from app.backend.utils.deep.dependencies import dependencies

config_model = {
    "caffe_path": "app/backend/model/face/res10_300x300_ssd_iter_140000.caffemodel",
    "config_path": "app/backend/model/face/deploy.prototxt.txt",
    "deep_model_path": [
        "app/backend/model/deep_model/main_model/model_last_minimum_CNN-bestSave_-distillation.hdf5",
        "app/backend/model/deep_model/main_model/_CNN.-0.72.hdf5",
        "app/backend/model/deep_model/main_model/model_last_model_3-100-70.hdf5"],
    "meta_model_path": "app/backend/model/deep_model/main_model/model_last_2_model_LogisticRegression_nohist.pkl",
    "dictionary": {
        "Angry": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happy": 3,
        "Sad": 4,
        "Surprise": 5,
        "Neutral": 6,
        "None": -1
    },
    "labels": ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    "dependencies": dependencies,
    "face_lib": {
        "scoreThreshold": 0.7,
        "iouThreshold": 0.8,
    }
}
