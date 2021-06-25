from .process.Segment import Segmentation
from fastapi.templating import Jinja2Templates
from app.backend.core import TEMPLATE



def render(image_path: str, request):
    return TEMPLATE.TemplateResponse(image_path, {"request": request})

def segment_image(image_path, segment_fn, request):
    name_path, mask = segment_fn(image_path)
    return {"image": name_path, "mask": mask}