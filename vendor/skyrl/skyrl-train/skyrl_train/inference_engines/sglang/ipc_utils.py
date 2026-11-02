"""Utilities for IPC request serialization for SGLang weight transfer."""

import pickle
import base64

import torch

from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest


IPC_REQUEST_END_MARKER = b"__END_OF_REQUEST__"


def serialize_ipc_request(request: NamedWeightsUpdateRequest) -> torch.Tensor:
    """Serialize a weight update request to a tensor for IPC transfer.

    Args:
        request: Weight update request dict.

    Returns:
        A uint8 tensor containing the serialized request.
    """
    request_data = pickle.dumps(request)
    request_data_encoded = base64.b64encode(request_data)
    data_with_marker = request_data_encoded + IPC_REQUEST_END_MARKER

    # Pad for 4-byte alignment
    data_size = len(data_with_marker)
    padded_size = ((data_size + 3) // 4) * 4
    tensor_data = bytearray(data_with_marker)
    tensor_data.extend(b"\x00" * (padded_size - data_size))
    return torch.frombuffer(tensor_data, dtype=torch.uint8)


def deserialize_ipc_request(tensor: torch.Tensor) -> NamedWeightsUpdateRequest:
    """Deserialize a weight update request from a tensor.

    Args:
        tensor: A uint8 tensor containing the serialized request.

    Returns:
        The deserialized weight update request dict.

    Raises:
        ValueError: If the tensor doesn't contain a valid serialized request.
    """
    tensor_bytes = tensor.cpu().numpy().tobytes()
    end_index = tensor_bytes.find(IPC_REQUEST_END_MARKER)
    if end_index == -1:
        raise ValueError("End marker not found in tensor data")
    request_data = tensor_bytes[:end_index]
    try:
        request_data_decoded = base64.b64decode(request_data)
        return pickle.loads(request_data_decoded)
    except Exception as e:
        raise ValueError("Failed to deserialize request data") from e
