from pydantic import BaseModel


class BaseSchemas(BaseModel):
    class Config():
        orm_mode = True