from pydantic import BaseModel


class TourismInRequest(BaseModel):
    ids: int = 10
    keyword: str = "Hello"
    title: str = "World"
    content: str = "Hello Worlds"
    source: str = "sawqr"


if __name__ == "__main__":
    a = TourismInRequest
    b = a(ids=3,
          keyword="hell",
          title="Worlds",
          content="a",
          source="aeqwe")
    print(b.ids)
