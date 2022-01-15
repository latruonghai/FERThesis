FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
WORKDIR /code

RUN apt-get update && \
    apt-get install -y git && \
    apt-get -y install cmake && \
    apt-get install -y wget unzip && \
    apt-get install -y --no-install-recommends software-properties-common libboost-all-dev libc6-dbg libgeos-dev python3.7 python3-pip python3-setuptools && \
    apt-get install -y libjpeg-dev zlib1g-dev && \
    apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    rm -rf /var/lib/apt/lists/*

COPY . /code/
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r requirements.txt && \
    mkdir app/backend/model/deep_model app/backend/model/deep_model/main_model &&\
    wget "https://drive.google.com/uc?export=download&id=1-J23m6p5sdXrwTzqaM8xAENODtIA4DJ1" -O "app/backend/model/deep_model/main_model/_CNN.-0.72.hdf5" &&\
    wget "https://drive.google.com/uc?export=download&id=10kCvU7vqc4AxX7j6rvAdswb0SQugBOil" -O "app/backend/model/deep_model/main_model/model_last_model_3-100-70.hdf5" && \
    wget "https://drive.google.com/uc?export=download&id=1-rvZEKqt6w88OOxmavXFHjudvoDEsw99" -O "app/backend/model/deep_model/main_model/model_last_2_model_LogisticRegression_nohist.pkl" && \
    wget "https://drive.google.com/uc?export=download&id=1-J23m6p5sdXrwTzqaM8xAENODtIA4DJ1" -O "app/backend/model/deep_model/main_model/model_last_minimum_CNN-bestSave_-distillation.hdf5"
RUN python3.7 -m pip install wrapt termcolor
CMD ["python3.7", "main.py"]