from .process.Segment import Segmentation
from fastapi.templating import Jinja2Templates

template = Jinja2Templates(directory="app/backend/jinja/templates")

def render(image_path: str, request):
    return template.TemplateResponse(image_path, {"request": request})
