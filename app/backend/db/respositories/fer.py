from fastapi.responses import JSONResponse
from app.backend.configs import config_model

from app.backend.utils.module_recognize.fer import FER
from tensorflow.compat.v1 import ConfigProto
from app.backend.model.schemas.FER import FERRequest
from tensorflow.compat.v1 import InteractiveSession

from app.backend.utils.module_recognize.fer import read_image_from_byte, decode_image, encode_image
# from app.backend.model.schemas.FER import FERRequest
# from app.backend.model.schemas.FER import


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

fer = FER(config_model)


def detect_face_with_image(file: FERRequest):

    image = decode_image(file.file)

    img, dic_face = fer.detect_emotion_with_image(
        image, gpu=int(file.gpu), show=bool(file.show))
    # print(dic_face)
    imgcode = encode_image(img)

    return JSONResponse(
        {
            "header": "Content-type: application/json",
            "body": {
                "state": 1 if dic_face else 0,
                "image": str(imgcode),
                "dict_face": dic_face}})
