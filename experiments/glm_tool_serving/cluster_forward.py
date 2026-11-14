# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward a pod-local inference listener to its cluster network interface."""

import asyncio
import sys


async def main(listen_port: int, target_port: int) -> None:
    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", target_port)

        async def copy(source: asyncio.StreamReader, target: asyncio.StreamWriter) -> None:
            try:
                while data := await source.read(65536):
                    target.write(data)
                    await target.drain()
            finally:
                target.close()

        await asyncio.gather(copy(reader, upstream_writer), copy(upstream_reader, writer))

    server = await asyncio.start_server(connection, "0.0.0.0", listen_port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
