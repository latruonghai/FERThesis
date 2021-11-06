import json


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