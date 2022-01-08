from pydantic import BaseModel


class FERRequest(BaseModel):
    gpu: int = -1
    show: bool = False
    file: str = ""
    process: int=0