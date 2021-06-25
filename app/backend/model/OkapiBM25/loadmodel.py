import pickle5 as pickle
from app.backend.core.Paths import file_path



def load_model(file_path):
    
    with open(file_path[0], 'rb') as f:
        avgdl = pickle.load(f)
    with open(file_path[1], 'rb') as f:
        dl = pickle.load(f)
    with open(file_path[2], 'rb') as f:
        dltable = pickle.load(f)
    with open(file_path[3], 'rb') as f:
        f2t = pickle.load(f)
    with open(file_path[4], 'rb') as f:
        files = pickle.load(f)
    with open(file_path[5], 'rb') as f:
        idf = pickle.load(f)
    with open(file_path[6], 'rb') as f:
        iDex = pickle.load(f)
    return (files, f2t, iDex, dltable, dl, avgdl, idf)



FILEPATH = load_model(file_path)