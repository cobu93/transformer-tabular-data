FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /app
COPY requirements.txt .

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y git
RUN pip install -r requirements.txt
# RUN pip install 