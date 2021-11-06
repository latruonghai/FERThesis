import json
import os
import operator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from face_lib import face_lib
from skimage.feature import hog
from skimage import exposure
import time
from sklearn.svm import SVC
import glob
from sklearn.metrics import accuracy_score
from tfcuda import session
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from numpy import dstack
from keras import backend as K
from keras.metrics import Accuracy
from sklearn.model_selection import GridSearchCV
import pickle
import random

# from sklearn.decomposition import PCA

svc = SVC(C=1, kernel='linear')

config = {
    "caffe_path": "model/face/res10_300x300_ssd_iter_140000.caffemodel",
    "config_path": "model/face/deploy.prototxt.txt",
    "deep_model_path": [
    "./model/model_last/model_last_vgg16-58.hdf5",
    "./model/model_last/CNN-99.hdf5"],
    "meta_model_path": "model/model_last/model_last_2_model_GridSearchCV_nohist_0.7300000190734863.pkl",
}
# print(svc)
data = [{
    "name": "svm_hog_size_48",
    "component_size": ((146, 48), (76, 48)),
    "model": svc,
    "hog_bins": 9,
    "hog_pixel_cell": [(16, 16), (8, 8), (4, 4)],
    "hog_cell_block": [(4, 4), (2, 2)],
    "accuracies": [],
    "name_to_save": [],
    "hog_models": [

    ],
    "best_hog": None
},
    {
    "name": "svm_hog_size_64",
    "component_size": ((194, 64), (102, 64)),
    "model": svc,
    "hog_bins": 9,
    "hog_pixel_cell": [(16, 16), (8, 8), (4, 4)],
    "hog_cell_block": [(4, 4), (2, 2)],
    "accuracies": [],
    "name_to_save": [],
    "hog_models": [

    ],
    "best_hog": None
},
    {
    "name": "svm_hog_size_96",
    "component_size": ((292, 96), (152, 48)),
    "model": svc,
    "hog_bins": 9,
    "hog_pixel_cell": [(16, 16), (8, 8), (4, 4)],
    "hog_cell_block": [(4, 4), (2, 2)],
    "accuracies": [],
    "name_to_save": [],
    "hog_models": [

    ],
    "best_hog": None
},
    {
    "name": "svm_hog_size_128",
    "component_size": ((388, 128), (204, 128)),
    "model": svc,
    "hog_bins": 9,
    "hog_pixel_cell": [(16, 16), (8, 8), (4, 4)],
    "hog_cell_block": [(4, 4), (2, 2)],
    "accuracies": [],
    "name_to_save": [],
    "hog_models": [

    ],
    "best_hog": None
},
]

base_hog_model = {
    "hog_bins": None,
    "hog_cell_block": None,
    "hog_pixel_cell": None,
    "accuracy": None

}

class MultiTrain:

    def __init__(self, data, json_file, base_dict):
        self._init(data, json_file, base_dict)

    def _init(self, data, json_file, base_dict):
        self.data = data
        self.proc = DatasetProcess()
        self.js = JsonReader()
        data = js.get_json_from(json_file)
        self.train_ = data["train"]
        self.test_ = data["test"]
        self.dic = None
        self.base_dict = base_dict

    def processing_dataset(self, eye_size, mouth_size,
                           hog_cell_block, hog_pixel_cell, hog_bins):
        """
        # Preprocessing Dataset: Using hog transform to normalize facial components include (eye, mouth)

        ---
        params:
            - eye_size (tuple): size of the eye component of face
            - mouth_size (tuple): size of the mouth component of face
            - hog_bins (int): bins of orientation in hog algorithm
            - hog_cell_block (tuple): size of cell per clock in an image
            - hog_pixel_cell (tuple): size of pixel per cell in an image

        ---
        return:
            - x_train, x_test, y_train, y_test: train, test dataset after preprocessing.
            With:
                - x_train: list of hog feature vector of train dataset
                - x_test: test dataset's list of hog feature vector of test dataset
                - y_train: train dataset's list of label
                - y_test: test dataset's list of label
        """

        x_train, y_train = self.proc.create_x_y(self.train_, save_data=False, visualize=False, quiet_mode=True,
                                                eye_size=eye_size, mouth_size=mouth_size, hog_bins=hog_bins,
                                                hog_cell_block=hog_cell_block, hog_pixel_cell=hog_pixel_cell)
        x_test, y_test = self.proc.create_x_y(self.test_, save_data=False, visualize=False, quiet_mode=True,
                                              eye_size=eye_size, mouth_size=mouth_size, hog_bins=hog_bins,
                                              hog_cell_block=hog_cell_block, hog_pixel_cell=hog_pixel_cell)

        return x_train, x_test, y_train, y_test

    def train(self, x_train, y_train, model):
        model.fit(x_train, y_train)
        return model

    def predict_score(self, model, x_test, y_test):
        y_pred = model.predict(x_test)

        return accuracy_score(y_test, y_pred)

    def update_base_dict(self, hog_bins, hog_cell_block,
                         hog_pixel_cell, accuracy):
        base_dict = self.base_dict.copy()
        base_dict["hog_bins"] = hog_bins
        base_dict["hog_cell_block"] = hog_cell_block
        base_dict["hog_pixel_cell"] = hog_pixel_cell
        base_dict["accuracy"] = accuracy

        return base_dict
#         return base_dict

    def __fit_one_data(self, data):
        data["accuracies"] = []
        model = data["model"]

        eye_size = data["component_size"][0]
        hog_bins = data["hog_bins"]
        mouth_size = data["component_size"][1]
        max_index, max_value = -1, 0
        index = 0
        for hog_cell_block in data["hog_cell_block"]:
            for hog_pixel_cell in data["hog_pixel_cell"]:

                print(f"Processing with:\nEye size:{eye_size}\tMouth size:{mouth_size}\nHog:\nHog bins:{hog_bins}\tCell per block: {hog_cell_block}\tPixel per cell: {hog_pixel_cell} ")
                x_train, x_test, y_train, y_test = self.processing_dataset(eye_size=eye_size, mouth_size=mouth_size, hog_cell_block=hog_cell_block, hog_pixel_cell=hog_pixel_cell, hog_bins=hog_bins)

                print(f"Training with model {str(model)}")
                model = self.train(x_train, y_train, model)
                print("Training Done!!\nConinue predicting")
                accuracy_scores = self.predict_score(model, x_test, y_test)
                if max_value < accuracy_scores:
                    max_value = accuracy_scores
                    max_index = index
                print(f"Accuracy score of this process is {accuracy_scores}\n")
                base_dict = self.update_base_dict(hog_bins, hog_cell_block, hog_pixel_cell, accuracy_scores)
                data["accuracies"].append(base_dict)
                print(data["accuracies"])
                index += 1
#         self.sort_accuracy()
        data["best_hog"] = data["accuracies"][max_index]
        print("Done")

    def fit(self):
        #         base_dict = base_hog_model
        for data in self.data:
            self.__fit_one_data(data)

    def _sort_accuracies(self, data):
        accuracy = data["accuracies"]
        values = [member["accuracy"] for member in accuracy]
        print(values)
#         values_max = max(values)
        max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))

        data["best_hog"] = accuracy[max_index]
#         index =
#         for dic in accuracy:

    def sort_accuracy(self):
        for data in self.data:
            self._sort_accuracies(data)

    def convert_one_to_json(self, data):
        name = data["name"]
        data["model"] = str(data["model"])
        path = f"../model/model best/{name}_svc.json"
        with open(path, "w") as f:
            json.dump(f, data)
            print("You have save the best description into {}".format(os.path.abspath(path)))

    def convert_to_json(self):
        for data in self.data:
            self.covert_one_to_json(data)
            

class ConvertJson:
    def __init__(self, path):
        self.dic = None
        self.path = path

    def get_name_list(self, url_name, os_name = "linux"):
#         print(url_name)
        return url_name.split("/")
    
    def read_image_from_folder(self):
        pass
    def to_json(self, name_file="temp.json"):
        self.read_image_from_folder()
        with open(name_file, "w") as file:
            json.dump(self.dic, file)
            print(f"You have saved the json file to {os.path.abspath(os.path.join('.', name_file))}")

# class ConvertJsonCKPlus(ConvertJson):
#     def __init__(self, path):
#         super().__init__(path)
        
#     def read_image_from_folder(self):
#         self.dic = {}
#         for folder, subfolde

class ConvertJsonJaffe(ConvertJson):
    def __init__(self, path):
        super().__init__(path)
        self.dic = {}
        self.kw = {"FE": "fear", "AN": "angry", "DI": "disgust",
                   "HA": "happy", "NE": "neutral", "SA": "sad", "SU": "surprised"}

    def __get_name_of_file___(self, file_name, index=0):
        return file_name.split(".")[index]

    def __get_keywords__(self, file_name):
        kws = self.__get_name_of_file___(file_name, index=1)[0:2]
        return self.kw[kws]

    def get_properties_from_kw(self, key, value):
        try:
            self.dic[key].append(value)
        except BaseException:
            self.dic[key] = []
            self.dic[key].append(value)

    def read_image_from_folder(self):
        self.dic = {}
#         print("Hello")
#         print(self.path)
        for path_file in glob.glob(self.path):
            #             print(path_file)
            name_list = self.get_name_list(path_file)
            key = self.__get_keywords__(name_list[-1])
            self.get_properties_from_kw(key, path_file)
            
            
class ConvertJsonFER(ConvertJson):
    
    def __init__(self, path):
        super().__init__(path)
    
    
    def read_image_from_folder(self):
        self.dic = {}
        for folder, subfolder, files in os.walk(self.path):
            if len(subfolder) == 0:
#                 print(folder)
                name_list = self.get_name_list(folder)
#                 print(name_list)
                csv_file_name = name_list[-2]
                properties = name_list[-1]
                if csv_file_name[0] != "." and csv_file_name[0]!="_":
                    if not csv_file_name in self.dic.keys():
                        self.dic[csv_file_name] ={}
                    for file in files:
                        full_path = os.path.join(folder, file)
                        try:
                            self.dic[csv_file_name][properties].append(full_path)
                        except BaseException:
                            self.dic[csv_file_name][properties] = []
                            self.dic[csv_file_name][properties].append(full_path)
    # Convert to csv
    
            
class JsonReader:

    def __init__(self):
        self.json = None

    def json_load(self, path):
        with open(path, "r") as f:
            self.json = json.load(f)

    def get_json_from(self, path):
        self.json_load(path)
        return self.json
    
class DatasetProcess:
    
    def __init__(self):
        pass
        
    def create_x_y(self, data, save_data=False, visualize=False, quiet_mode=True,
                    hog_bins=9, hog_pixel_cell=(16,16), hog_cell_block=(2, 2), 
                    eye_size=(146,48), mouth_size = (76,48)):
        x_train, y_train = [], []
        det = Detect()
        for key, value in data.items():
            for val in value:
                try:
                    hog_vec = det.detect_component(val, save_data=save_data, visualize=visualize, quiet_mode=quiet_mode,
                                                  hog_bins=hog_bins, hog_pixel_cell=hog_pixel_cell, hog_cell_block=hog_cell_block, 
                                                    eye_size=eye_size, mouth_size =mouth_size)
                except:
#                     raise BaseException("There are some error")
                    continue
                y_train.append(key)
#                 img_value = cv2.imread(val, cv2.IMREAD_GRAYSCALE)
#                 print(img_value.shape)
                x_train.append(hog_vec)
        print("Done, there is {} in this dataset".format(len(y_train)))
        return np.array(x_train), np.array(y_train)

# from utils import ConvertJson, ConvertJsonFER, JsonReader, DatasetProcess


class PlotImage:

    def __init__(self):
        self.fig = None

    def __check_isPrime__(self, num):
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def factorial(self, num):
        fac = {}
        for i in range(2, num + 1):
            while self.__check_isPrime__(i) and (num % i == 0):
                if i not in fac.keys():
                    fac[i] = 1
#                     num//=i
                else:
                    fac[i] += 1
#                     num//=i
                num //= i
        return fac

    def get_true_index(self, max_row, row, col):
        return row * max_row + col + row

    def plot(self, image,max_row, max_col, labels=None, name_to_save=None):
        if len(image)>10:
            new_image = image[0:6]
            if labels!=None:
                new_labels = labels[0:6]
        else:
            new_image=image
            new_labels=labels
#         if self.__check_isPrime__(num_of_image):
#             num_of_image += 1
        self.fig, self.ax = plt.subplots(max_row, max_col, figsize=(10, 5))
        for row in range(len(self.ax)):
            try:
                for col in range(len(self.ax[row])):
                    true_index = self.get_true_index(max_row, row, col)

                    self.ax[row][col].axis("off")
                    self.ax[row][col].imshow(new_image[true_index], cmap=plt.cm.gray)
                    if labels!=None:
                        self.ax[row][col].set_title(f"Image {true_index}: {new_labels[true_index]}")
                    else:
                        self.ax[row][col].set_title(f"Image {true_index}")
            except:
                true_index = self.get_true_index(max_row, 0, row)
                self.ax[row].axis("off")
                self.ax[row].imshow(new_image[true_index])
                if labels!=None:
                    self.ax[row].set_title(f"Image {true_index}: {new_labels[true_index]}")
                else:
                    self.ax[row].set_title(f"Image {true_index}")
        if name_to_save:
            self.save_plt(name_to_save)
                #         fig, ax = plt.subplot(
    def save_plt(self, name_to_save="temp.jpg"):
        try:
            self.fig.savefig(name_to_save, bbox_inches='tight', dpi=150)
        
            plt.close(self.fig)
        except:
            pass
                

# Name Preprocessing
class Preprocessing:
    def __init__(self):
        self.kw = {"AF": "afraid", "AN": "angry", "DI": "disgust",
                   "HA": "happy", "NE": "neutral", "SA": "sad", "SU": "surprised"}

    def get_name_list(self, path):
        return path.split("/")

    def get_name_image_file(self, list_name):
        return list_name[-1]

    def get_name_image(self, image_file: str):
        return image_file.split(".")[0]

    def get_emotion_from_key(self, name_image: str):
        kws = name_image[4:6]
        return self.kw[kws]

    def get_name_emotion_from(self, path):

        name_list = self.get_name_list(path)
        name_image_file = self.get_name_image_file(name_list)
        name_image = self.get_name_image(name_image_file)
        return self.get_emotion_from_key(name_image)


class PathProcessing:
    def get_full_path(self, rela_path):
        return os.path.abspath(rela_path)

    def read_path(self, path):
        return os.walk(path)

    def join_path(self, *path):
        return os.path.join(path)


class GetDataJsonKDEF:
    def __init__(self, path_folder="."):
        self.dict = {"KDEF dataset": {

        }}
        self.minidict = self.dict["KDEF dataset"]
        self.path = PathProcessing()
        self.path_folder = self.path.read_path(path_folder)
        self.preprocess = Preprocessing()

    def add_value_to_key(self, key, value):
        try:
            self.minidict[key].append(value)
        except BaseException:
            self.minidict[key] = []
            self.minidict[key].append(value)

    def read_image_in_paths(self):
        for folder, subfolder, files in self.path_folder:
            if len(subfolder) == 0:
                for file in files:
                    full_path = os.path.join(folder, file)
#                     print(full_path)
                    emotion = self.preprocess.get_name_emotion_from(full_path)
                    self.add_value_to_key(emotion, full_path)

    def save_to_json(self, root_path=".", name_to_save="temp.json"):
        full_path = os.path.join(root_path, name_to_save)
        with open(full_path, "w") as file:
            json.dump(self.dict, file)
        print("Your file have save in {}".format(self.path.get_full_path(full_path)))

    def get_json_from_image_path(self, name_file):
        self.read_image_in_paths()
        self.save_to_json(name_to_save=name_file)
        
class Coordinate:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.length = y + h
        self.wide = x + w

    def get_coordinate(self):
        return(self.x, self.y, self.wide, self.length)


def _calculate(eye_list):
    if len(eye_list) < 2:
        raise Exception("There should be 2 eyes, but have {} eyes detected".format(len(eye_list)))
    else:
        eye_1, eye_2 = eye_list[0:2]
        w3 = Coordinate(min(eye_1.x, eye_2.x), max(eye_1.y, eye_2.y), max(eye_2.wide, eye_1.wide) - min(eye_1.x, eye_2.x), max(eye_1.h, eye_2.h))
        return w3


def _get_coord(coord):
    return coord.get_coordinate()

class Detect:
    def __init__(self):
        self.eye_cascade_ = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect_face(self, img, save=False, name_to_save="./temp/temp.jpg", 
                    eye_size=(146,48), mouth_size=(76,48)):

        new_img = img.copy()
        faces = self.face_cascade.detectMultiScale(img)
#         print(len(faces))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 print("X, y, w, h", x, y, w, h)
                new_img = new_img[y:y + h, x:x + w]
                img_new = img[y:y + h, x:x + w]
#                 print("IMG Shape", img_new.shape)
#                 plt.imshow(img_new)

#                 img_new = cv2.resize(img_new, (128, 128))
#                 new_img = cv2.resize(new_img, (128, 128*new_img.shape[1]//new_img.shape[0]))
                img, face_img, eye_region = self.detect_eye(img_new, new_img, eye_size=eye_size)
                img, mouth_img = self.mouth_region(img, eye_region, mouth_size=mouth_size)
#                 print(face_img)
                break
#                 img, new_mouth_img = self.detect_mouth(img_new, new_img)
            return img, cv2.GaussianBlur(face_img,(5,5),cv2.BORDER_DEFAULT), cv2.GaussianBlur(mouth_img,(5,5),cv2.BORDER_DEFAULT)
        else:
            raise Exception("Could not find face")

    def detect_eye(self, ori_img, face_img, eye_size=(146,48)):
        new_face_img = face_img.copy()
#         print(new_face_img)
        eyes_list = []
        eyes = self.eye_cascade_.detectMultiScale(face_img)
        for (x, y, w, h) in eyes:
#             cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             print("X,y,w,H", x, y, w, h)
            eyes_list.append(Coordinate(x, y, w, h))

        new_eye = _calculate(eyes_list)
        X, Y, X2, Y2 = _get_coord(new_eye)
#         print(x, y, x2, y2)
        cv2.rectangle(ori_img, (X, Y), (X2, Y2), (0, 255, 255), 4)

        new_face_img = new_face_img[Y:Y2, X:X2]
#         print(new_face_img.shape)
        new_face_img = cv2.resize(new_face_img, (388, 128))
        return ori_img, new_face_img, new_eye

    def mouth_region(self, ori_img, eye_region, mouth_size=(76,48)):
        new_mouth = ori_img.copy()
        x, y, x2, y2 = _get_coord(Coordinate(eye_region.x, eye_region.length + 3, eye_region.w, ori_img.shape[1] - eye_region.length))
        cv2.rectangle(ori_img, (x, y), (x2, y2), (0, 0, 255), 2)
        new_mouth = new_mouth[y:y2, x:x2]
#         print(new_mouth.shape)
        new_mouth = cv2.resize(new_mouth, (204, 128))
        return ori_img, new_mouth
        #     def __concat_list__(self, *args):
        #         list_ = []
        #         for arg in args:
        #             try:
        #                 list_ += [ar for ar in arg]
        #             except BaseException:
        #                 list_.append(arg)
        #         return list_

    def detect_component(self, path_img, visualize=False, save_data=False, quiet_mode=True,
                                hog_bins=9, hog_pixel_cell=(16,16), hog_cell_block=(2, 2), 
                                 eye_size=(146,48), mouth_size = (76,48)):
        img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.equalizeHist(img)
#         clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
#         img = clahe.apply(img)
#         img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
#         img = cv2.resize(img, (224, int(224 * img.shape[1] / img.shape[0])))
        img, face_img, mouth_img = self.detect_face(img, eye_size=eye_size, mouth_size=mouth_size)
#         print(face_img.shape)
#         list_img = selflf.__concat_list__(img, face_img)

#     + [mouth for mouth in mouth_img]
#         print(face_img[0].shape)
        fd, hog_image = hog(face_img, orientations=hog_bins, pixels_per_cell=hog_pixel_cell,
                            cells_per_block=hog_cell_block, visualize=True)
#         hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        fd, hog_image_mouth = hog(mouth_img, orientations=9, pixels_per_cell=(16, 16),
                                  cells_per_block=(2, 2), visualize=True)
#         hog_image = cv2.GaussianBlur(hog_image,(5,5),cv2.BORDER_DEFAULT)
#         hog_image_mouth = cv2.GaussianBlur(hog_image_mouth,(5,5),cv2.BORDER_DEFAULT)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        hog_image_mouth_rescaled = exposure.rescale_intensity(hog_image_mouth, in_range=(0, 10))
        
        list_img = [img, face_img, mouth_img, hog_image_rescaled, hog_image_mouth_rescaled]
#        
        hog_image_mouth_rescaled = hog_image_mouth_rescaled.reshape(-1)
        hog_image_rescaled = hog_image_rescaled.reshape(-1)
#         print(hog_image_mouth_rescaled.shape)
#         print(hog_image_rescaled.shape)
        new_hog = np.concatenate((hog_image_rescaled, hog_image_mouth_rescaled), axis=0)
#         pca = PCA()
#         pca.fit(new_hog)
#         new_hog = pca.singular_values_
        pl = PlotImage()
#         print(new_hog.shape)
        if visualize:
            
            pl.plot(list_img, 1, len(list_img), labels=None)
        if save_data:
            name_list = path_img.split("/")
#             print(path_img)
            emotion_folder = name_list[-2]
            file_name = name_list[-1]
            train_test = name_list[-3]
            full_path = os.path.join("map", train_test, emotion_folder)
            try:
                os.mkdir(full_path)
            except:
                pass
            full_path = os.path.join(full_path, file_name)
            full_path = os.path.abspath(full_path)
#             print(full_path)
            pl.save_plt(name_to_save=full_path)
            if not quiet_mode:
                print(f"You have save data to image in {os.path.join(full_path, file_name)}")
        return new_hog
    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save=True,
                          name_to_save="temp.jpg"):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save:
        plt.savefig(name_to_save)
    plt.show()

    


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

class DetectorModule:
    """
        Detect face with different modules include: SSD, Viola & Jones, FaceLibs.
        
        ---
        Attributes:
            - face_cascade: module for Viola & Jone detector module
            - FL: moudle for face_lib detector module
            - net: module for SSD detector module
        ---
        Methods:
            - _get_base_detector: load base model for SSD detector module
            - set_parms_FL: set default parameters for face lib module
            - detect_with_VJ: detect face using Viola & Jones algorithm
            - detect_with_FL: detect face using Facelib model
            - detect_with_gpu: detect face using SSD model
            - detect_face: detect face using the combine of three module include: VJ, FL, GPU with selector params
        Params:
        
            - config (dict): configuration for detector Module
        ---
        Return:
            - faces (coordinate) for each module
    """
    def __init__(self, config):

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.FL = face_lib()
        self.set_params_FL(scoreThreshold=0.85, iouThreshold=0.7)
        self.net = self._get_base_detector(config)
        
    def set_params_FL(self, scoreThreshold=0.85, iouThreshold=0.7):
        self.FL.set_detection_params(scoreThreshold, iouThreshold)
        print(f"You have change Facelib's params into Score Threshold: {scoreThreshold} and IoU Threshold: {iouThreshold}")
        
    def _get_base_detector(self, config):
        modelFile = config["caffe_path"]
        configFile = config["config_path"]
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        print("Loaded caffe and config file")
        return net
    
    def detect_with_VJ(self, img_new):
        """
            Detect Face using Viola & Jones algorithm
            ---
            Params: 
                - img_new (numpy array): array include pixel from gray scale image
            
            ---
            Return:
                - faces: (list): list of coordination for each face in faces list are x, y, w, h (x, y, width, height) respectively.
        """
        faces = self.face_cascade.detectMultiScale(img_new, minNeighbors=5,
		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return faces
    def detect_with_FL(self ,image):
        
        """
            Detect Face using Facelib model
            
            Params:
            --------
                - image(numpy array): array include pixel from gray scale image 
            
            Return:
            --------
                - no_of_faces (int): number of faces
                - face_coors: coordinate of face are x, y, w, h (x, y, width, height) respectively.
        """
        no_of_faces, faces_coors =  self.FL.faces_locations(image, max_no_faces=20)
        return no_of_faces, faces_coors
    def detect_with_gpu(self, img):
        """
            Detect Face using Facelib model
            
            Params:
            --------
                - img(numpy array): array include pixel from gray scale image 
            
            Return:
            --------
                - faces: (list): list of coordination for each face in faces list are x, y, w, h (x, y, width, height) respectively.
                - h, w: coordinate of image to fit to official image are height, width respectively.
        """
        #     print(img.shape)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        return faces, h, w
    
    def detect_face(self,image, gpu=1):
        if gpu==1:
            return self.detect_with_gpu(image)
        elif gpu == 0:
            return self.detect_with_VJ(image)
        else:
            return self.detect_with_FL(image)
#             return face

class FER:
    
    def __init__(self, config):
        
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.detector_module = DetectorModule(config)
        self.enmodel = EnsembleModel(config)
    def set_params_FL(self, scoreThreshold=0.85, iouThreshold=0.7):
        
        self.detector_module.set_params_FL(scoreThreshold, iouThreshold)

    def face_recognition_with_gpu(self, img, quiet=True):
        # Face detection with dnn
        start = time.time()
        faces, h, w = self.detector_module.detect_face(img, gpu=1)
        end = time.time() - start
        if not quiet:
            print(f"GPU Detected in {end}s")
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_image = img_new[y:y1, x:x1]
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                try:
                    face_image = cv2.resize(face_image, (48, 48))
                    face_image = face_image / 127.5
                    face_image -= 1
                    face_image = np.expand_dims(np.array([face_image]), -1)
                except BaseException:
                    continue
                start = time.time()
                pred_label, pred_score = self.enmodel.stacked_prediction(face_image)
                
    #             print(f'Predict in {time.time()-start}s')
                label = self.labels[pred_label[0]]
                cv2.putText(img, f'{label}: {pred_score}', (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(250,191,163) , 2)
                end2 = time.time() - start
            else:
                continue
#         print(img.shape)
        return img, end, end2

    def face_recognition_with_face_lib(self, image, quiet=True):
#         print("use lib")
        img_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start = time.time()
        no_of_faces, faces_coors = self.detector_module.detect_face(image, gpu=-1)
        end = time.time() - start
        if not quiet:
            print(f"Face Lib Detected in {end}s")
        if no_of_faces >0:
#             print(no_of_faces)
            for x, y, w, h in faces_coors:
#                 print(x,y,w,h)
                face_image = img_new[y:y + h, x:x + w]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                face_image = cv2.resize(face_image, (48, 48))
                face_image = face_image / 127.5
                face_image -= 1
                face_image = np.expand_dims(np.array([face_image]), -1)
                start = time.time()
                pred, score = self.enmodel.stacked_prediction(face_image)
                #             print(f'Predict in {time.time()-start}s')
                label = self.labels[pred[0]]
                cv2.putText(image, f'{label}:{score}', (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (216,103,90), 2)
                end2 = time.time() - start
        else:
            raise BaseException("There are no face's detected")
        return image, end, end2
    def face_recognition_with_cpu(self, image, quiet = False ):
        # Face detection with haarcascade

        img_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_new = cv2.equalizeHist(img_new)
        start = time.time()
        faces = self.detector_module.detect_face(img_new, gpu=0)
        end = time.time() - start
        if not quiet:
            print(f"CPU Detected in {end}s")
        # img_new=Nonez
        if len(faces) > 0:
            for x, y, w, h in faces:
                face_image = img_new[y:y + h, x:x + w]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                face_image = cv2.resize(face_image, (48, 48))
                face_image = face_image / 127.5
                face_image -= 1
                face_image = np.expand_dims(np.array([face_image]), -1)
                start = time.time()
                pred, score = self.enmodel.stacked_prediction(face_image)
    #             print(f'Predict in {time.time()-start}s')
                label = self.labels[pred[0]]
                cv2.putText(image, f'{label}:{score}', (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,153,67), 2)
                end2 = time.time() - start

        else:
            raise BaseException("There are no face's detected")
        return image, end, end2
    def processing_image(self, img, gpu=1, quiet=False, resize = True):
        #     img = cv2.imread(path_image)
        if isinstance(img, str):
            img = cv2.imread(img)
        if resize:
            img = cv2.resize(img, (400, 400 * img.shape[0] // img.shape[1]))
        try:
            if gpu ==1:
                face_image, end, end2 = self.face_recognition_with_gpu(img, quiet)
            elif gpu ==0:
                face_image, end, end2 = self.face_recognition_with_cpu(img, quiet)
            else:
                face_image, end, end2 = self.face_recognition_with_face_lib(img, quiet)
                
            return face_image, end, end2
        except:
            return img  # cv2_imshow(img)
        
    def save_img(self, face, file_name="temp.jpg"):
        file_name = file_name.split('.')[0]
        path_save = f'{file_name}_result1.jpg'
        cv2.imwrite(path_save, face)
        print(f'You saved images to the path {path_save}')
        
    def detect_emtion_with_image(self, img_path, gpu=1, save=True):
        if gpu == 1:
            print("You are using gpu")
        elif gpu == 0:
            print("You are using cpu")
        else:
            print("You are using Face Lib")
        face, _, __ = self.processing_image(img_path, gpu)
        cv2.imshow("Frame", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save:
            self.save_img(face, img_path)
    
    def detect_emotion_with_vid(self, vid_path=0, gpu=True, quiet=True):
        if gpu == 1:
            print("You are using gpu")
        elif gpu == 0:
            print("You are using cpu")
        else:
            print("You are using Face Lib")
        vid = cv2.VideoCapture(vid_path)
        frame_rate = 10
        prev = 0
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('output.mp4', fourcc, 7, (534, 400))
        while(True):

            ret, frame = vid.read()
            if ret:

                prev = time.time()
            #     faces = face_cascade.detectMultiScale()
                frame = cv2.resize(frame, (400, 400 * frame.shape[0] // frame.shape[1]))
                # Display the resulting frame
                try:
                    face_emotion, end, end2 = self.processing_image(frame, gpu, quiet, resize = False)
#                     print(end, end2)
                    frame = face_emotion
                    fps1 = self.calculate_fps(end)
                    fps2 = self.calculate_fps(end2)
#                     print("Fps:",fps1, fps2)
                    cv2.putText(frame, f'Fps Detector:{fps1}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (144,65,67), 2)
                    cv2.putText(frame, f'Fps Emotion:{fps2}', (frame.shape[1]-150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (144,65,67), 2)
            #             pass
                except BaseException:
                    pass
                
                cv2.imshow('frame', frame)
        #         out.write(frame)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()
        # out.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
    def get_base_detector(self, config):
        modelFile = config["caffe_path"]
        configFile = config["config_path"]
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        print("Loaded caffe and config file")
        return net
    def calculate_fps(self, end):
        fps = 1/end
        fps = str(int(np.ceil(fps)))

        return fps
class EnsembleModel:
    def __init__(self, config):
        n_models_path = config["deep_model_path"]
        print(n_models_path)
        meta_model_path = config["meta_model_path"]
        self.dependencies = {"accuracy": Accuracy, "f1_m": f1_m}
        self.members = self.load_all_models(n_models_path)
        
        self.model = self.load_meta_model(meta_model_path)
        print(f'Load {len(self.members)} Deep models')
        
        
    def load_all_models(self, n_models_path):
        all_models = list()
        for path in n_models_path:
#             print(path)
            # Specify the filename
            # filename = '/content/model' + str(i + 1) + '.h5'
            # load the model
            try:
                model = load_model(path, custom_objects=self.dependencies)
            except BaseException as err:
                print("Cant Load Model and ", err)
                return None
                # Add a list of all the weaker learners
            all_models.append(model)
            print('>loaded %s' % path)
        return all_models
    def load_meta_model(self, meta_model_path):
#         print(meta_model_path)
        with open(meta_model_path, 'rb') as f:
            model_ = pickle.load(f)
        print("Meta Model Was Loaded")
#         print(str(model_))
        return model_
    
    def stacked_dataset(self,inputX):
        stackX = None
        for model in self.members:
            # make prediction
            yhat = model.predict(inputX, verbose=0)
            # stack predictions into [rows, members, probabilities]
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))
        # flatten predictions to [rows, members x probabilities]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
        return stackX

    # Fit a model based on the outputs from the ensemble members


    def fit_stacked_model(self, inputX, inputy):
        if not self.model:
            # create dataset using ensemble
            stackedX = self.stacked_dataset(inputX)
            print(stackedX.shape, inputy.shape)
            # fit standalone model
            inputy = np.argmax(inputy, axis=1)
            model = SVC()  # meta learner
        #   parameters = {'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'C':[1, 10, 5, 15], 'tol':[0.001, 0.0001, 0.0005]}
        #   clf = GridSearchCV(model, parameters, return_train_score=True)
            model.fit(stackedX, inputy)
        #   print(clf.best_estimator_)
            self.model = model
        else:
            print("Model was created")
#         return model
    
    def stacked_prediction(self, inputX):
        # create dataset using ensemble
        stackedX = self.stacked_dataset(inputX)
        # make a prediction
        # model.probability=True
        pred = self.model.predict_proba(stackedX)
    #     print("Probs:", pred)
        yhat = np.argmax(pred, axis=1)
        # print()
    #     print(yhat)
        # probabilities = np.array(list(map(predict_prob, yhat)))
        # print(probabilities)
        return yhat, np.round(np.max(pred, axis=1) * 100, 2)[0]


    # Evaluate model on test set

    def predict_with_model(self, inputx, inputy):
        yhat = self.stacked_prediction(members, model, inputx)
        # print(yhat.shape)
        # yhat = convert_to_onehot(yhat)
        # yval =
        yval_temp = np.argmax(inputy, axis=1)

        return yhat, yval_temp
    

class SplitImage:
    def __init__(self, root = "model/picture/test/result"):
        
        self.root = root
        self.tiles = None
#         fontScale = min(imageWidth,imageHeight)/(25/scale)
#     def set_params(self, random, )
    def split(self, im_path, times:tuple, save=True, israndom = True, putText =True):
        if isinstance(im_path, str):
            im = cv2.imread(im_path)
        M = im.shape[0] // times[0]
        N = im.shape[1] // times[1]
        self.tiles = [im[x:x + M, y:y + N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
        if israndom:
            random.shuffle(self.tiles)
        if save:
            self.save_image(im_path, putText = putText)
        return self.tiles
    def save_image(self, img_path, putText = True):
        filename = img_path.split("/")[-1].split(".")[0]
        full_path = self.create_folder(self.root, filename)
        for index, tile in enumerate(self.tiles):
            # print(tile.shape)
            lowerLeftTextOriginX, lowerLeftTextOriginY, resultText, fontScale = self.set_font_scale(tile, index)
            if putText:
                cv2.putText(tile, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (50, 250, 163), 5)
            path_file = os.path.join(full_path, f"result-{filename}-{index}.jpg")
            cv2.imwrite(path_file, tile)
        print(f'Folder {full_path} done. There are {index+1} picture in this folder')
    def get_optimal_font_scale(self, text, width):
#         width = tile.shape[1]
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
            new_width = textSize[0][0]
          # print(new_width)
            if (new_width <= width):
                return scale / 10
        return 1
    def set_font_scale(self,tile, index, scale=1):
        
        resultText = str(index)
        imageHeight, imageWidth, _ = tile.shape
        fontScale = self.get_optimal_font_scale(resultText, imageWidth)
        upperLeftTextOriginX = int(imageWidth * 0.05)
        upperLeftTextOriginY = int(imageHeight * 0.05)

        textSize, baseline = cv2.getTextSize(resultText, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 5)
        textSizeWidth, textSizeHeight = textSize

            # calculate the lower left origin of the text area based on the text area center, width, and height
        lowerLeftTextOriginX = upperLeftTextOriginX
        lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight
        
        return lowerLeftTextOriginX, lowerLeftTextOriginY, resultText, fontScale//2
    def create_folder(self, root, subfolder):
        full_path = os.path.join(root, subfolder)
        try:
            os.mkdir(full_path)
            print(f"Folder {full_path} was created")
        except FileExistsError:
            
            print(f"Folder {full_path} existed, deleting folder")
            import shutil
            shutil.rmtree(f'{full_path}')
            print("Deleting Done")
            return self.create_folder(root, subfolder)
        return full_path
if __name__ =="__main__":
    js = ConvertJson(".")
    js.to_json(name_file="FER_temp.json")
    
    
    