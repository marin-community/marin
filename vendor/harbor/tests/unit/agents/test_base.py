import pytest

from harbor.agents.base import MalformedModelResponseError, encode_model_output


def test_encode_model_output_returns_utf8_bytes():
    assert encode_model_output("café 🦖") == b"caf\xc3\xa9 \xf0\x9f\xa6\x96"


def test_encode_model_output_classifies_surrogate_as_model_failure():
    text = "echo \ud842\n"

    with pytest.raises(MalformedModelResponseError) as caught:
        encode_model_output(text)

    assert isinstance(caught.value.__cause__, UnicodeEncodeError)
    assert text.rstrip() in caught.value.__cause__.object
