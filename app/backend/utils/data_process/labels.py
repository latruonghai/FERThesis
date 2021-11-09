# from keras.metrics import Accuracy

class TranslateLabel:
    def __init__(self, config):
        self.dictionary = config["dictionary"]

    def translate(self, label_array):
        if isinstance(label_array, str):
            new_label = self.dictionary[label_array]
        else:
            #             new_label = self.dictionary
            unique_name = np.unique(label_array)
            new_label = label_array.copy()
            for name in list(unique_name):
                new_label[label_array == name] = self.dictionary[name]
        return new_label
# def validate_(y_pred, y_true):
#         score = Accuracy()
#         y_pred = np.argmax(y_pred, axis=1)
#         y_true = np.argmax(y_true, axis=1)
#         score.update_state(y_pred, y_true)
#         print(score.result().numpy())

#         return score.result().numpy()


class LabelProcessing:
    def __init__(self):
        #         self.y_true = None
        pass

    def label_dummies(self, true_label):
        #         true_label = np.array(label * numOfSample)
        y_true = pd.get_dummies(true_label).to_numpy()
        print(y_true.shape)

        return y_true

#     def get_y_pred(self, numpy_file):
#         y_pred = np.array([])
#         for i in range(1, len(numpy_file)):
#             y_pred = np.concatenate((y_pred, numpy_file[i, 2:]))
# #         print(y_pred)
#         y_pred = pd.get_dummies(y_pred).to_numpy()
#         print(y_pred.shape)
#         return y_pred

    def label_one_hot_encode(self, true_label):
        true_label = self.label_dummies(true_label)
        return np.argmax(true_label, axis=1)
