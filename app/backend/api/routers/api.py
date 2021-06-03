from fastapi import APIRouter
from . import images

router = APIRouter()
router.include_router(images.router, tags=["Images"])
@router.get("/")
async def hello_world():
    return {"title": "Hello",
            "body": "What's up"}
