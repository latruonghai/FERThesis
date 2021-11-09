from app.backend.configs import config_model
from app.backend.utils.module_recognize.fer import FER
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

fer = FER(config_model)


def detect_face_with_image(img, gpu, show):
    return fer.detect_emotion_with_image(img, gpu=gpu, show=show)
