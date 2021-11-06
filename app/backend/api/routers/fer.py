from fastapi import APIRouter, File, UploadFile
from starlette.responses import StreamingResponse
from app.backend.db.respositories.fer import detect_face_with_image
from app.backend.utils.module_recognize.fer import read_image_from_byte
import numpy as np
from PIL import Image
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, imencode
import base64
import io

router = APIRouter()

@router.post("/fer/recognition/image")
# def recog
async def api_recognize(file: UploadFile = File(...), gpu=1, show=False):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    
    if not extension:
        return "Image must be jpg or png format"
    pic = read_image_from_byte(await file.read())
    image = np.array(pic)
    image = cvtColor(image, COLOR_RGB2BGR)
    # print(image.shape)
    img = detect_face_with_image(image, gpu=int(gpu), show=show)
    # img = cvtColor(img, COLOR_BGR2RGB)
    # print(img.shape)
    # im_pil = Image.fromarray(img)
    # encode = base64.b64decode(img)
    
    res, im_png = imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    # return im_pil

    # print(img)