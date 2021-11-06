import cv2
from face_lib import face_lib


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

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.FL = face_lib()
        self.set_params_FL(scoreThreshold=0.85, iouThreshold=0.7)
        self.net = self._get_base_detector(config)

    def set_params_FL(self, scoreThreshold=0.85, iouThreshold=0.7):
        self.FL.set_detection_params(scoreThreshold, iouThreshold)
        print(
            f"You have change Facelib's params into Score Threshold: {scoreThreshold} and IoU Threshold: {iouThreshold}")

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
        faces = self.face_cascade.detectMultiScale(
            img_new, minNeighbors=5, minSize=(
                30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    def detect_with_FL(self, image):
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
        no_of_faces, faces_coors = self.FL.faces_locations(
            image, max_no_faces=20)
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

    def detect_face(self, image, gpu=1):
        if gpu == 1:
            return self.detect_with_gpu(image)
        elif gpu == 0:
            return self.detect_with_VJ(image)
        else:
            return self.detect_with_FL(image)
