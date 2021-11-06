import os


class Preprocessing:
    def __init__(self):
        self.kw = {
            "AF": "afraid",
            "AN": "angry",
            "DI": "disgust",
            "HA": "happy",
            "NE": "neutral",
            "SA": "sad",
            "SU": "surprised"}

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
