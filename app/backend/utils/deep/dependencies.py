from keras.metrics import Accuracy
from .validation import f1_m
from adabelief_tf import AdaBeliefOptimizer

adabelief = AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-7, rectify=False,print_change_log = False)

dependencies = {
    "accuracy": Accuracy,
    "AdaBeliefOptimizer": adabelief,
    "f1_m": f1_m}
