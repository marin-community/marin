# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed OpenBao KV-v2 bindings, each read through its own native credential."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rigging.openbao import OpenBaoClient


@dataclass(frozen=True)
class SecretBinding:
    name: str
    mount: str
    path: str
    address: str

    @property
    def resource(self) -> str:
        return f"/secrets/{self.name}/"

    @property
    def api_path(self) -> str:
        return f"/{self.mount}/data/{self.path}"


class SecretConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    mount: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    path: str = Field(pattern=r"^[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*$")
    token_file: Path

    def binding(self, address: str) -> SecretBinding:
        return SecretBinding(self.name, self.mount, self.path, address)


@dataclass(frozen=True)
class OpenBaoSecretReader:
    binding: SecretBinding
    client: OpenBaoClient

    def read(self) -> dict:
        response = self.client.request("GET", self.binding.api_path)
        data = response["data"]["data"]
        if not isinstance(data, dict):
            raise ValueError("The secret provider returned malformed KV data")
        return data
