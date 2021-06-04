from .process.Segment import Segmentation
from fastapi.templating import Jinja2Templates
from app.backend.core import TEMPLATE



def render(image_path: str, request):
    return TEMPLATE.TemplateResponse(image_path, {"request": request})
