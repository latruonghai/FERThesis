from fastapi import APIRouter, File, UploadFile
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse
from app.backend.db.respositories.fer import detect_face_with_image
from app.backend.utils.module_recognize.fer import read_image_from_byte, decode_image, encode_image
from app.backend.model.schemas.FER import FERRequest
# from app.backend.model.schemas.FER import
import numpy as np
from PIL import Image

import base64
import io

router = APIRouter()


@router.post("/fer/recognition/image", response_class=JSONResponse)
# def recog
async def api_recognize(file: FERRequest):
    return detect_face_with_image(file)
    

