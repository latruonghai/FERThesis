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
    def __init__(self, config):
        n_models_path = config["deep_model_path"]
        print(n_models_path)
        meta_model_path = config["meta_model_path"]
        self.dependencies = config["dependencies"]
        self.members = self.load_all_models(n_models_path)
        
        self.model = self.load_meta_model(meta_model_path)
        print(f'Load {len(self.members)} Deep models')
        
        
    def load_all_models(self, n_models_path):
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
    def load_meta_model(self, meta_model_path):
#         print(meta_model_path)
        with open(meta_model_path, 'rb') as f:
            model_ = pickle.load(f)
        print("Meta Model Was Loaded")
#         print(str(model_))
        return model_
    
    def stacked_dataset(self,inputX):
        stackX = None
        for model in self.members:
            # make prediction
            yhat = model.predict(inputX, verbose=0)
            # stack predictions into [rows, members, probabilities]
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))
        # flatten predictions to [rows, members x probabilities]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
        return stackX

    # Fit a model based on the outputs from the ensemble members


    def fit_stacked_model(self, inputX, inputy):
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
        yhat = self.stacked_prediction(members, model, inputx)
        # print(yhat.shape)
        # yhat = convert_to_onehot(yhat)
        # yval =
        yval_temp = np.argmax(inputy, axis=1)

        return yhat, yval_temp
