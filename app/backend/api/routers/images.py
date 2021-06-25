from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from app.backend.db.respositories import images
from fastapi.templating import Jinja2Templates
from app.backend.model.CV.segment import Segmentation
from app.backend.core.Paths import path_model

router = APIRouter()
SEG = Segmentation(path_model)

@router.get('/images', response_class=HTMLResponse)
def get(request: Request):
    return images.render("CV.html", request)


@router.get('/image/segment', response_class=JSONResponse)
def segment(name_picture, request: Request):
    full_path = f"./app/backend/jinja/static/image/image_before/{name_picture}"
    return images.segment_image(image_path=full_path, segment_fn=SEG.Processing, request=request)