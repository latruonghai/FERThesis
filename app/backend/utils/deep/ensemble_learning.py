import numpy as np
from keras.models import load_model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from numpy import dstack
import time
# from keras.metrics import Accuracy
# from .validation import f1_m
import pickle


class EnsembleModel:
    """
    Using Ensemble method (stacking) to predict labels for facial image. This module is base
    model that predicts facial emotion
    ----
    Attributes:
        - config: (dictionary): including configure for that ensemble method. There are: dependencies
    model path (deep model path for loading model by tensorflow), meta model path (meta model load for Ensemble Model),

    ----

    Methods:
        - load_all_models -> array of model: for loading deep learning model
        - load_meta_model -> pickle model: for loading pickle model (meta model)
        - stacked_dataset -> numpy.stack: stack of predicted label from deep learning member models
        - fit_stacked_model -> sklearn.model: for training meta model based on stacked dataset.
        - stacked_prediction -> tuple: for predicting label from input based on Ensemble method.
        - predict_with_model -> tuple: predict model

    """

    def __init__(self, config):
        n_models_path = config["deep_model_path"]
        print(n_models_path)
        meta_model_path = config["meta_model_path"]
        self.dependencies = config["dependencies"]
        self.members = self.load_all_models(n_models_path)

        self.model = self.load_meta_model(meta_model_path)
        print(f'Load {len(self.members)} Deep models')

    def load_all_models(self, n_models_path: list):
        """
        Loading deep learning model. Using keras load model from list of model paths.

        Args:
            n_models_path (list): list of deep learning model's path, which were trained before.

        Returns:
            list: list of model loaded from keras.load_model method
        """

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

    def load_meta_model(self, meta_model_path: str):
        """
        Load Meta model for ensemble method

        Args:
            meta_model_path (str): meta model's path for loading meta model in ensemble method.

        Returns:
            pickle: loaded meta model.
        """
        #         print(meta_model_path)
        with open(meta_model_path, 'rb') as f:
            model_ = pickle.load(f)
        print("Meta Model Was Loaded")
#         print(str(model_))
        return model_

    def deep_predict(self, model_index=0, inputX=None):
        s = time.time()
        pred = self.members[model_index].predict(inputX)
        start = time.time() - s
        yhat = np.argmax(pred, axis=1)

        return yhat, np.round(np.max(pred, axis=1) * 100, 2)[0], start

    def stacked_dataset(self, inputX):
        """
        Stack of predicted label from deep learning member models

        Args:
            inputX (array): Input data (images).

        Returns:
            np.stack: stack of labels predicted from model in members (list of deep learning member)
        """
        stackX = None
        for model in self.members[1:]:
            # make prediction
            yhat = model.predict(inputX, verbose=0)
            # stack predictions into [rows, members, probabilities]
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))
        # flatten predictions to [rows, members x probabilities]
        stackX = stackX.reshape(
            (stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
        return stackX

    # Fit a model based on the outputs from the ensemble members

    def fit_stacked_model(self, inputX, inputy):
        """
        Train meta model from stack dataset created by deep learning models.

        Args:
            inputX (np.array): Input data
            inputy (np.array): Output data (ground-truth label)
        """
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
        """
        Predicting the label for input using ensemble method

        Args:
            inputX (np.array): Input data

        Returns:
            yhat (np.array): predicted label from inputX
            confidences (np.int16): confidence of yhat label
            end (float): processing time
        """
        # create dataset using ensemble
        start = time.time()
        stackedX = self.stacked_dataset(inputX)
        # make a prediction
        # model.probability=True

        pred = self.model.predict_proba(stackedX)
        end = time.time() - start
    #     print("Probs:", pred)
        yhat = np.argmax(pred, axis=1)
        # print()
    #     print(yhat)
        # probabilities = np.array(list(map(predict_prob, yhat)))
        # print(probabilities)
        return yhat, np.round(np.max(pred, axis=1) * 100, 2)[0], end

    # Evaluate model on test set

    def predict_with_model(self, inputx, inputy):
        """
        Predict Labels using ensemble method and standardize them
        Args:
            inputx (np.array): Input data
            inputy (np.array): Output data

        Returns:
            yhat: (np.array): Predicted labels
            yval_temp: (np.array): Labels after standardize
        """
        yhat = self.stacked_prediction(members, model, inputx)
        # print(yhat.shape)
        # yhat = convert_to_onehot(yhat)
        # yval =
        yval_temp = np.argmax(inputy, axis=1)

        return yhat, yval_temp
