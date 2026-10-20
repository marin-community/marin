//! Bounded zstd decoding without repeatedly growing large RPC body buffers.

use std::io::Read;

use bytes::Bytes;
use connectrpc::compression::{CompressionProvider, CompressionRegistry};
use connectrpc::ConnectError;

pub(super) fn compression_registry() -> CompressionRegistry {
    // Replacing the buffered provider retains the stock streaming providers.
    CompressionRegistry::default().register(ZstdProvider::default())
}

#[derive(Default)]
struct ZstdProvider {
    inner: connectrpc::compression::ZstdProvider,
}

impl CompressionProvider for ZstdProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn compress(&self, data: &[u8]) -> Result<Bytes, ConnectError> {
        self.inner.compress(data)
    }

    fn decompressor<'a>(&self, data: &'a [u8]) -> Result<Box<dyn Read + 'a>, ConnectError> {
        self.inner.decompressor(data)
    }

    fn decompress_with_limit(&self, data: &[u8], max_size: usize) -> Result<Bytes, ConnectError> {
        let size = zstd::zstd_safe::get_frame_content_size(data)
            .ok()
            .flatten()
            .and_then(|size| usize::try_from(size).ok())
            .filter(|size| *size <= max_size);
        if let Some(size) = size {
            // A first-frame size is not the total size of concatenated frames.
            if zstd::zstd_safe::find_frame_compressed_size(data).ok() == Some(data.len()) {
                return zstd::bulk::decompress(data, size)
                    .map(Bytes::from)
                    .map_err(|error| {
                        ConnectError::internal(format!("zstd decompression failed: {error}"))
                    });
            }
        }
        self.inner.decompress_with_limit(data, max_size)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use connectrpc::ErrorCode;
    use tokio::io::{AsyncReadExt, BufReader};

    use super::*;

    #[test]
    fn zstd_preserves_frames_and_aggregate_limits() {
        let body = b"log line\n".repeat(128);
        let known = zstd::bulk::compress(&body, 1).unwrap();
        let unknown = zstd::stream::encode_all(body.as_slice(), 1).unwrap();
        let skippable = [0x50, 0x2a, 0x4d, 0x18, 4, 0, 0, 0, 1, 2, 3, 4];
        let cases = [
            (known.clone(), body.clone()),
            (unknown, body.clone()),
            (
                [known.as_slice(), known.as_slice()].concat(),
                body.repeat(2),
            ),
            (
                [skippable.as_slice(), known.as_slice()].concat(),
                body.clone(),
            ),
            ([known.as_slice(), skippable.as_slice()].concat(), body),
            (skippable.to_vec(), Vec::new()),
            (zstd::bulk::compress(&[], 1).unwrap(), Vec::new()),
        ];
        let provider = ZstdProvider::default();
        for (compressed, expected) in cases {
            assert_eq!(
                provider
                    .decompress_with_limit(&compressed, expected.len())
                    .unwrap()
                    .as_ref(),
                expected
            );
            if !expected.is_empty() {
                assert_eq!(
                    provider
                        .decompress_with_limit(&compressed, expected.len() - 1)
                        .unwrap_err()
                        .code,
                    ErrorCode::ResourceExhausted
                );
            }
        }
    }

    #[test]
    fn invalid_zstd_preserves_stock_errors() {
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 1).unwrap();
        encoder.include_checksum(true).unwrap();
        encoder.set_pledged_src_size(Some(3)).unwrap();
        encoder.write_all(b"abc").unwrap();
        let valid = encoder.finish().unwrap();
        let mut bad_checksum = valid.clone();
        *bad_checksum.last_mut().unwrap() ^= 1;
        // Single-segment raw block with a one-byte advertised content size.
        let raw = [
            0x28, 0xb5, 0x2f, 0xfd, 0x20, 3, 0x19, 0, 0, b'a', b'b', b'c',
        ];
        let mut too_small = raw.to_vec();
        too_small[5] = 2;
        let mut too_large = raw.to_vec();
        too_large[5] = 255;
        let stock = connectrpc::compression::ZstdProvider::default();
        let provider = ZstdProvider::default();
        for data in [
            b"bad".to_vec(),
            valid[..valid.len() - 1].to_vec(),
            [valid.as_slice(), b"junk"].concat(),
            bad_checksum,
            too_small,
            too_large,
        ] {
            for limit in [3, 512] {
                assert_eq!(
                    provider
                        .decompress_with_limit(&data, limit)
                        .unwrap_err()
                        .code,
                    stock.decompress_with_limit(&data, limit).unwrap_err().code
                );
            }
        }
    }

    #[tokio::test]
    async fn registry_preserves_gzip_streaming_and_empty_bodies() {
        let registry = compression_registry();
        let body = b"log line\n".repeat(128);
        for encoding in ["gzip", "zstd"] {
            let compressed = registry.compress(encoding, &body).unwrap();
            assert_eq!(
                registry
                    .decompress_with_limit(encoding, compressed, body.len())
                    .unwrap()
                    .as_ref(),
                body
            );
            assert_eq!(
                registry
                    .decompress_with_limit(encoding, Bytes::new(), 0)
                    .unwrap(),
                Bytes::new()
            );
            let source = Box::pin(BufReader::new(Cursor::new(body.clone())));
            let mut compressed = registry.compress_stream(encoding, source).unwrap();
            let mut bytes = Vec::new();
            compressed.read_to_end(&mut bytes).await.unwrap();
            assert_eq!(
                registry
                    .decompress_with_limit(encoding, bytes.into(), body.len())
                    .unwrap()
                    .as_ref(),
                body
            );
        }
    }
}
