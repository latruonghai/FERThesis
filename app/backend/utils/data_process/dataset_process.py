from .face_detect.svm_detect import Detect

class DatasetProcess:
    """
    [summary]
    """
    def __init__(self):
        pass

    def create_x_y(
        self,
        data,
        save_data=False,
        visualize=False,
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
            data ([type]): [description]
            save_data (bool, optional): [description]. Defaults to False.
            visualize (bool, optional): [description]. Defaults to False.
            quiet_mode (bool, optional): [description]. Defaults to True.
            hog_bins (int, optional): [description]. Defaults to 9.
            hog_pixel_cell (tuple, optional): [description]. Defaults to ( 16, 16).
            hog_cell_block (tuple, optional): [description]. Defaults to ( 2, 2).
            eye_size (tuple, optional): [description]. Defaults to ( 146, 48).
            mouth_size (tuple, optional): [description]. Defaults to ( 76, 48).
        """
        x_train, y_train = [], []
        det = Detect()
        for key, value in data.items():
            for val in value:
                try:
                    hog_vec = det.detect_component(
                        val,
                        save_data=save_data,
                        visualize=visualize,
                        quiet_mode=quiet_mode,
                        hog_bins=hog_bins,
                        hog_pixel_cell=hog_pixel_cell,
                        hog_cell_block=hog_cell_block,
                        eye_size=eye_size,
                        mouth_size=mouth_size)
                except BaseException:
                    #                     raise BaseException("There are some error")
                    continue
                y_train.append(key)
#                 img_value = cv2.imread(val, cv2.IMREAD_GRAYSCALE)
#                 print(img_value.shape)
                x_train.append(hog_vec)
        print("Done, there is {} in this dataset".format(len(y_train)))
        
        return np.array(x_train), np.array(y_train)
