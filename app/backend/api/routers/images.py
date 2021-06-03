from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from app.backend.db.respositories import images
from fastapi.templating import Jinja2Templates

router = APIRouter()


@router.get('/images', response_class=HTMLResponse)
def get(request: Request):
    return images.render("CV.html", request)