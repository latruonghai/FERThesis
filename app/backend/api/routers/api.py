from fastapi import APIRouter
from . import fer

router = APIRouter()
router.include_router(fer.router, tags=["Recognition"])
# router.include_router(images.router, tags=["Images"])
# router.include_router(retrieval.router, tags=["IR"])
#router.include_router(hellman.router, tags=["hellman"])
@router.post("/")
async def hell(text: str):
    return {
        "title": "Hello",
        "text": text
    }

@router.get("/")
async def hello_world():
    return {"title": "Hello",
            "body": "What's up"}

