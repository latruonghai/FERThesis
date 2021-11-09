from app.backend.utils.deep.dependencies import dependencies

config_model = {
    "caffe_path": "app/backend/model/face/res10_300x300_ssd_iter_140000.caffemodel",
    "config_path": "app/backend/model/face/deploy.prototxt.txt",
    "deep_model_path": [
        "app/backend/model/model_new_last/model_last_vgg16-0.hdf5",
        "app/backend/model/model_new_last/_CNN.21-0.69.hdf5"],
    "meta_model_path": "app/backend/model/model_new_last/model_last_2_model_SVC_nohist_0.7200000286102295.pkl",
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
    "face_lib":{
        "scoreThreshold": 0.5,
        "iouThreshold": 0.9,
    }
}
