from time import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import io
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os


class Segmentation():

    def __init__(self, model_path):
        # self.image_path = image_path
        # print("Will create model Seg")
        self.model_path = model_path

    def Processing(self, image_path):
        model = LoadModel(self.model_path)
        # img = cv2.imread(self.image_path)
        #print("Memory: ", self.GetSMIInfor())
        return self.Prediction(image_path, model)

        # cuda.get_current_device().reset()

    def Prediction(self, img_path, model):
        X = np.empty((1, 256, 256, 3))
        name = img_path.split('/')[-1].split('.')[0]
        img = io.imread(img_path)
        # print("Path:", img_path)
        img = cv2.resize(img, (256, 256))
        new_img = img.copy()
        img = np.array(img, dtype=np.float64)
        img -= img.mean()
        img /= img.std()
        X[0, ] = img
        predict = model.predict(X)
        predict = predict.reshape((256, 256)).astype(np.int64)

        # print("Shape: ", predict[predict > 0].shape)
        # img[predict == 0] = (255, 0 ,0)
        if predict.round().astype(int).sum() == 0:
            # image_id.append(i)
            # has_mask.append(0)
            # mask.append('No mask :)')
            # th, im_th = cv2.threshold(predict, 200, 255, cv2.THRESH_BINARY_INV)

            # contours, hierarchy = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask = 0
            # image = cv2.drawContours(predict, contours, -1, (0, 255, 0), 2)
            # y_tru.append(y_true)
            return name + ".jpg"
        else:
            # if the sum of pixel values are more than 0, then there is tumour
            # image_id.append(i)
            # has_mask.append(1)
            # mask.append(predict)
            mask = 1
            predict = predict.round().astype(int)
            predict[predict != 0] = 255
            cv2.imwrite("temp.jpg", predict)
            predict = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
            # predict = cv2.cvtColor(predict, cv2.COLOR_RGB2GRAY)
            th, im_th = cv2.threshold(predict, 200, 255, cv2.THRESH_BINARY_INV)

            contours, hierarchy = cv2.findContours(
                im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            image = cv2.drawContours(predict, contours, -1, (0, 255, 0), 2)

            # cv2_imshow(predict)
            # io.imsave(f"/content/drive/MyDrive/AdvanceCV/y_pred/{name}.jpg",predict)
            # io.imsave(f"/content/drive/MyDrive/AdvanceCV/y_test/{name}.jpg", y_true)
            new_img[image > 0] = (122, 12, 1)
            full_path = f"./app/backend/jinja/static/image/image_after/{name}.jpg"
            io.imsave(full_path, new_img)
        print("Ban da format thanh cong")
        return (name + ".jpg", mask)

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
        "focal_tversky_loss": focal_tversky_loss,
        "tversky": tversky,
        "tversky_loss": tversky_loss,
        "diceBCE_loss": diceBCE_loss,
        "iou": iou,
        "dice_coef": dice_coef
    })


def tversky(y_true, y_pred):
    epsilon = 1e-5
    smooth = 1

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha *
                                  false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return 1 - tversky(y_true, y_pred)


def iou(y_true, y_pred):
    smooth = 1e-6
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    intersection = K.sum(inputs * targets)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    return (intersection + smooth) / (union + smooth)


def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    return 1 - iou(y_true, y_pred)


def diceBCE_loss(y_true, y_pred):
    smooth = 1e-6

    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    dice_coef_sc = dice_coef_loss(y_true, y_pred)
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)

    return BCE + dice_coef_sc


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.cast(y_pred, tf.float32)
    return -dice_coef(y_true, y_pred)


def ShowInfor(img):
    img = io.imread(img)
    #img = np.array(img)
    print(img[img == 0].shape)
