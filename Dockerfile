# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:slim

# EXPOSE 8000

# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED=1

# # Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt


# WORKDIR /app
# COPY . /app

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app && \
#     wget "https://drive.google.com/uc?export=download&id=1-J23m6p5sdXrwTzqaM8xAENODtIA4DJ1" -O "app/backend/model/deep_model/main_model/_CNN.-0.72.hdf5" &&\
#     wget "https://drive.google.com/uc?export=download&id=10kCvU7vqc4AxX7j6rvAdswb0SQugBOil" -O "app/backend/model/deep_model/main_model/model_last_model_3-100-70.hdf5" && \
#     wget "https://drive.google.com/uc?export=download&id=1-rvZEKqt6w88OOxmavXFHjudvoDEsw99" -O "app/backend/model/deep_model/main_model/model_last_2_model_LogisticRegression_nohist.pkl"
# USER appuser

# # During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python3", "-m", "main.py"]

FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

WORKDIR /backend

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y git && \
    apt-get -y install cmake && \
    apt-get install -y wget unzip && \
    apt-get install -y --no-install-recommends software-properties-common libboost-all-dev libc6-dbg libgeos-dev python3-dev python3-pip python3-setuptools && \
    apt-get install -y libjpeg-dev zlib1g-dev && \
    apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    rm -rf /var/lib/apt/lists/*

COPY . /backend/
RUN python3 -m pip install --upgrade pip
RUN ls && pip3 install -r requirements.txt

# CMD [ "executable" ]

