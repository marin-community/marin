from pydantic import BaseModel


class OpenAIChatMessage(BaseModel):
    role: str
    content: str
