from .coordinate import Coordinate
import cv2
from app.backend.utils.plot import PlotImage
from skimage.feature import hog
from skimage import exposure

class Detect:
    """[summary]
    """
    def __init__(self):

        self.eye_cascade_ = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect_face(self, img, save=False, name_to_save="./temp/temp.jpg",
                    eye_size=(146, 48), mouth_size=(76, 48)):
        """[summary]

        Args:
            img ([type]): [description]
            save (bool, optional): [description]. Defaults to False.
            name_to_save (str, optional): [description]. Defaults to "./temp/temp.jpg".
            eye_size (tuple, optional): [description]. Defaults to (146, 48).
            mouth_size (tuple, optional): [description]. Defaults to (76, 48).
        """
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
                img, face_img, eye_region = self.detect_eye(
                    img_new, new_img, eye_size=eye_size)
                img, mouth_img = self.mouth_region(
                    img, eye_region, mouth_size=mouth_size)
#                 print(face_img)
                break
#                 img, new_mouth_img = self.detect_mouth(img_new, new_img)
            return img, cv2.GaussianBlur(
                face_img, (5, 5), cv2.BORDER_DEFAULT), cv2.GaussianBlur(
                mouth_img, (5, 5), cv2.BORDER_DEFAULT)
        else:
            raise Exception("Could not find face")

    def detect_eye(self, ori_img, face_img, eye_size=(146, 48)):
        """[summary]

        Args:
            ori_img ([type]): [description]
            face_img ([type]): [description]
            eye_size (tuple, optional): [description]. Defaults to (146, 48).
        """
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

    def mouth_region(self, ori_img, eye_region, mouth_size=(76, 48)):
        """[summary]

        Args:
            ori_img ([type]): [description]
            eye_region ([type]): [description]
            mouth_size (tuple, optional): [description]. Defaults to (76, 48).
        """
        new_mouth = ori_img.copy()
        x, y, x2, y2 = _get_coord(
            Coordinate(
                eye_region.x, eye_region.length + 3, eye_region.w, ori_img.shape[1] - eye_region.length))
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

    def detect_component(
        self,
        path_img,
        visualize=False,
        save_data=False,
        quiet_mode=True,
        hog_bins=9,
        hog_pixel_cell=(
            16,
            16),
        hog_cell_block=(
            2,
            2),
            eye_size=(
                146,
                48),
        mouth_size=(
            76,
            48)):
        """[summary]

        Args:
            path_img ([type]): [description]
            visualize (bool, optional): [description]. Defaults to False.
            save_data (bool, optional): [description]. Defaults to False.
            quiet_mode (bool, optional): [description]. Defaults to True.
            hog_bins (int, optional): [description]. Defaults to 9.
            hog_pixel_cell (tuple, optional): [description]. Defaults to ( 16, 16).
            hog_cell_block (tuple, optional): [description]. Defaults to ( 2, 2).
            eye_size (tuple, optional): [description]. Defaults to ( 146, 48).
            mouth_size (tuple, optional): [description]. Defaults to ( 76, 48).
        """
        img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.equalizeHist(img)
#         clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
#         img = clahe.apply(img)
#         img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
#         img = cv2.resize(img, (224, int(224 * img.shape[1] / img.shape[0])))
        img, face_img, mouth_img = self.detect_face(
            img, eye_size=eye_size, mouth_size=mouth_size)
#         print(face_img.shape)
#         list_img = selflf.__concat_list__(img, face_img)

#     + [mouth for mouth in mouth_img]
#         print(face_img[0].shape)
        fd, hog_image = hog(face_img, orientations=hog_bins, pixels_per_cell=hog_pixel_cell,
                            cells_per_block=hog_cell_block, visualize=True)
#         hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        fd, hog_image_mouth = hog(
            mouth_img, orientations=9, pixels_per_cell=(
                16, 16), cells_per_block=(
                2, 2), visualize=True)
#         hog_image = cv2.GaussianBlur(hog_image,(5,5),cv2.BORDER_DEFAULT)
#         hog_image_mouth = cv2.GaussianBlur(hog_image_mouth,(5,5),cv2.BORDER_DEFAULT)
        hog_image_rescaled = exposure.rescale_intensity(
            hog_image, in_range=(0, 10))

        hog_image_mouth_rescaled = exposure.rescale_intensity(
            hog_image_mouth, in_range=(0, 10))

        list_img = [
            img,
            face_img,
            mouth_img,
            hog_image_rescaled,
            hog_image_mouth_rescaled]
#
        hog_image_mouth_rescaled = hog_image_mouth_rescaled.reshape(-1)
        hog_image_rescaled = hog_image_rescaled.reshape(-1)
#         print(hog_image_mouth_rescaled.shape)
#         print(hog_image_rescaled.shape)
        new_hog = np.concatenate(
            (hog_image_rescaled, hog_image_mouth_rescaled), axis=0)
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
            except BaseException:
                pass
            full_path = os.path.join(full_path, file_name)
            full_path = os.path.abspath(full_path)
#             print(full_path)
            pl.save_plt(name_to_save=full_path)
            if not quiet_mode:
                print(
                    f"You have save data to image in {os.path.join(full_path, file_name)}")
        return new_hog
