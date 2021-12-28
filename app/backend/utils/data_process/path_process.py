import os


class Preprocessing:
    """[summary]
    """
    def __init__(self):
        self.kw = {
            "AF": "afraid",
            "AN": "angry",
            "DI": "disgust",
            "HA": "happy",
            "NE": "neutral",
            "SA": "sad",
            "SU": "surprised"}

    def get_name_list(self, path: str):
        """[summary]

        Args:
            path (str): [description]

        Returns:
            [type]: [description]
        """
        return path.split("/")

    def get_name_image_file(self, list_name: list):
        """[summary]

        Args:
            list_name (list): [description]

        Returns:
            [type]: [description]
        """
        return list_name[-1]

    def get_name_image(self, image_file: str):
        """[summary]

        Args:
            image_file (str): [description]

        Returns:
            [type]: [description]
        """
        return image_file.split(".")[0]

    def get_emotion_from_key(self, name_image: str):
        """[summary]

        Args:
            name_image (str): [description]

        Returns:
            [type]: [description]
        """
        kws = name_image[4:6]
        return self.kw[kws]

    def get_name_emotion_from(self, path: str):
        """[summary]

        Args:
            path (str): [description]

        Returns:
            [type]: [description]
        """
        name_list = self.get_name_list(path)
        name_image_file = self.get_name_image_file(name_list)
        name_image = self.get_name_image(name_image_file)
        return self.get_emotion_from_key(name_image)


class PathProcessing:
    """[summary]
    """
    def get_full_path(self, rela_path: str):
        """[summary]

        Args:
            rela_path (str): [description]

        Returns:
            [type]: [description]
        """
        return os.path.abspath(rela_path)

    def read_path(self, path: str):
        """[summary]

        Args:
            path (str): [description]

        Returns:
            [type]: [description]
        """
        return os.walk(path)

    def join_path(self, *path: str):
        """[summary]

        Returns:
            [type]: [description]
        """
        return os.path.join(path)
