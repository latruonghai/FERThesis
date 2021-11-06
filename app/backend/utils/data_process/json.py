import json
import os


class ConvertJson:
    def __init__(self, path):
        self.dic = None
        self.path = path

    def get_name_list(self, url_name, os_name="linux"):
        #         print(url_name)
        return url_name.split("/")

    def read_image_from_folder(self):
        pass

    def to_json(self, name_file="temp.json"):
        self.read_image_from_folder()
        with open(name_file, "w") as file:
            json.dump(self.dic, file)
            print(
                f"You have saved the json file to {os.path.abspath(os.path.join('.', name_file))}")

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
        self.kw = {
            "FE": "fear",
            "AN": "angry",
            "DI": "disgust",
            "HA": "happy",
            "NE": "neutral",
            "SA": "sad",
            "SU": "surprised"}

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
                if csv_file_name[0] != "." and csv_file_name[0] != "_":
                    if csv_file_name not in self.dic.keys():
                        self.dic[csv_file_name] = {}
                    for file in files:
                        full_path = os.path.join(folder, file)
                        try:
                            self.dic[csv_file_name][properties].append(
                                full_path)
                        except BaseException:
                            self.dic[csv_file_name][properties] = []
                            self.dic[csv_file_name][properties].append(
                                full_path)
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
        print("Your file have save in {}".format(
            self.path.get_full_path(full_path)))

    def get_json_from_image_path(self, name_file):
        self.read_image_in_paths()
        self.save_to_json(name_to_save=name_file)
