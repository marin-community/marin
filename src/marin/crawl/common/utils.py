import logging

from resiliparse.parse.encoding import bytes_to_str, detect_encoding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def decode_html(html: bytes) -> str | None:
    """
    Given HTML (bytes), decode it into a string if possible. First try with
    utf-8. If that doesn't work, try to detect the encoding.
    """
    try:
        html = bytes_to_str(html, "utf-8")
    except Exception:
        encoding = detect_encoding(html)
        if encoding is None or encoding == "utf-8":
            return
        try:
            html = bytes_to_str(html, encoding)
        except Exception:
            logger.error(f"Failed to decode HTML with encoding {encoding}")
            return
    return html
