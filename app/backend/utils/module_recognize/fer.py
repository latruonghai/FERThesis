
# from app.backend.utils.face_detect import DetectorModule
# from app.backend.utils.deep import EnsembleModel
import cv2
import numpy as np
import time
from PIL import Image
from io import BytesIO
from app.backend.utils.face_detect.detector import DetectorModule
from app.backend.utils.deep.ensemble_learning import EnsembleModel




class FER:

    def __init__(self, config):

        self.labels = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Neutral',
            'Sad',
            'Surprise']
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
                pred_label, pred_score = self.enmodel.stacked_prediction(
                    face_image)

    #             print(f'Predict in {time.time()-start}s')
                label = self.labels[pred_label[0]]
                cv2.putText(
                    img,
                    f'{label}: {pred_score}',
                    (x - 2,
                     y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (250,
                     191,
                     163),
                    2)
                end2 = time.time() - start
            else:
                continue
#         print(img.shape)
        return img, end, end2

    def face_recognition_with_face_lib(self, image, quiet=True):
        #         print("use lib")
        img_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start = time.time()
        no_of_faces, faces_coors = self.detector_module.detect_face(
            image, gpu=-1)
        end = time.time() - start
        if not quiet:
            print(f"Face Lib Detected in {end}s")
        if no_of_faces > 0:
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
                cv2.putText(
                    image,
                    f'{label}:{score}',
                    (x - 2,
                     y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (216,
                     103,
                     90),
                    2)
                end2 = time.time() - start
        else:
            raise BaseException("There are no face's detected")
        return image, end, end2

    def face_recognition_with_cpu(self, image, quiet=False):
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
                cv2.putText(
                    image,
                    f'{label}:{score}',
                    (x - 2,
                     y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (225,
                     153,
                     67),
                    2)
                end2 = time.time() - start

        else:
            raise BaseException("There are no face's detected")
        return image, end, end2

    def processing_image(self, img, gpu=1, quiet=False, resize=True):
        #     img = cv2.imread(path_image)
        if isinstance(img, str):
            img = cv2.imread(img)
        if resize:
            img = cv2.resize(img, (400, 400 * img.shape[0] // img.shape[1]))
        try:
            if gpu == 1:
                face_image, end, end2 = self.face_recognition_with_gpu(
                    img, quiet)
            elif gpu == 0:
                face_image, end, end2 = self.face_recognition_with_cpu(
                    img, quiet)
            else:
                face_image, end, end2 = self.face_recognition_with_face_lib(
                    img, quiet)

            return face_image, end, end2
        except BaseException as er:
            print("Error:", er)
            return img, None, None  # cv2_imshow(img)

    def save_img(self, face, file_name="temp.jpg"):
        file_name = file_name.split('.')[0]
        path_save = f'{file_name}_result1.jpg'
        cv2.imwrite(path_save, face)
        print(f'You saved images to the path {path_save}')

    def detect_emotion_with_image(self, img_path, gpu=1, save=False, show=False):
        if gpu == 1:
            print("You are using gpu")
        elif gpu == 0:
            print("You are using cpu")
        else:
            print("You are using Face Lib")
        face, _, __ = self.processing_image(img_path, gpu)
        if show:
            cv2.imshow("Frame", face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            self.save_img(face, img_path)
            
        return face

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
                frame = cv2.resize(
                    frame, (400, 400 * frame.shape[0] // frame.shape[1]))
                # Display the resulting frame
                try:
                    face_emotion, end, end2 = self.processing_image(
                        frame, gpu, quiet, resize=False)
#                     print(end, end2)
                    frame = face_emotion
                    fps1 = self.calculate_fps(end)
                    fps2 = self.calculate_fps(end2)
#                     print("Fps:",fps1, fps2)
                    cv2.putText(
                        frame,
                        f'Fps Detector:{fps1}',
                        (20,
                         20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (144,
                         65,
                         67),
                        2)
                    cv2.putText(
                        frame,
                        f'Fps Emotion:{fps2}',
                        (frame.shape[1] -
                         150,
                         20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (144,
                         65,
                         67),
                        2)
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
        fps = 1 / end
        fps = str(int(np.ceil(fps)))

        return fps

def read_image_from_byte(image_data) -> Image.Image:
    image = Image.open(BytesIO(image_data))
    return image