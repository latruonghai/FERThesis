from app.backend.configs import config_model
from app.backend.utils.module_recognize.fer import FER

fer = FER(config_model)


def detect_face_with_image(img, gpu, show):
    return fer.detect_emotion_with_image(img, gpu, show=show)
