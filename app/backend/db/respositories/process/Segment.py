from time import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import io
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
import subprocess as sp


class Segmentation():

    def __init__(self, image_path, model_path):
        self.image_path = image_path
        # print("Will create model Seg")
        self.model_path = model_path

    def Processing(self, image_path):
        model = LoadModel(self.model_path)
        # img = cv2.imread(self.image_path)
        print("Memory: ", self.GetSMIInfor())
        return self.Prediction(image_path, model)

        # cuda.get_current_device().reset()

    def Prediction(self, img_path, model):
        X = np.empty((1, 256, 256, 3))
        name = img_path.split('/')[-1].split('.')[0]
        img = io.imread(img_path)
        #print("Path:", img_path)
        img = cv2.resize(img, (256, 256))
        new_img = img.copy()
        img = np.array(img, dtype=np.float64)
        img -= img.mean()
        img /= img.std()
        X[0, ] = img
        predict = model.predict(X)
        predict = predict.reshape((256, 256)).astype(np.int64)

        #print("Shape: ", predict[predict > 0].shape)
        # img[predict == 0] = (255, 0 ,0)
        new_img[predict > 0] = (122, 12, 1)
        io.imsave("/media/lahai/DATA/Study/query1/lib/Flask/application/" +
                  f'static/image/{name}.jpg', new_img)
        print("Ban da format thanh cong")
        return name + ".jpg"

    def GetModelInfo(self):
        return self.model

    def GetSMIInfor(self):

        def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(
            sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0])
                              for i, x in enumerate(memory_free_info)]
        return memory_free_values


def LoadModel(model_dir: str):
    return load_model(model_dir, custom_objects={
        "focal_tversky": focal_tversky,
        "tversky": tversky,
        "tversky_loss": tversky_loss
    })


def tversky(y_true, y_pred):

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7

    return (true_pos + 1) / (true_pos + alpha *
                             false_neg + (1 - alpha) * false_pos + 1)


def focal_tversky(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def ShowInfor(img):
    img = io.imread(img)
    #img = np.array(img)
    print(img[img == 0].shape)


class Convert():

    def __init__(self, dir_path):
        self.dir_path = dir_path
        print(self.dir_path)

    def GetConvert(self, img_path):
        path = os.path.join(self.dir_path, img_path)
        print(path)
        img = io.imread(path)
        img = cv2.resize(img, (256, 256))
        name = img_path.split(".")[0]
        full_name = f"/media/lahai/DATA/Study/query1/lib/Flask/application/static/image/job_conver/{name}.jpg"
        io.imsave(full_name, img)
        print(f'You have convert to {full_name}')

    def GetConvertFull(self):
        for f in os.listdir(self.dir_path):

            self.GetConvert(f)


if __name__ == '__main__':
    ''' con = Convert(
        "/media/lahai/DATA/Study/query1/lib/Flask/application/static/image/job")
    con.GetConvertFull() '''
    model_path = "/media/lahai/DATA/Study/query1/lib/Flask/model/weight/seg_model.h5"
    img_path = "/media/lahai/DATA/Study/query1/lib/Flask/application/static/image/job_conver/TCGA_CS_4942_19970222_9.jpg"
    # img_path1 = "/media/lahai/DATA/Study/query1/lib/Flask" + \
    #    "/application/static/image/Hello.tif"
    pass
