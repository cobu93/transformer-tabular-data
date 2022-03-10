FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y git
RUN pip install -r requirements.txt
# RUN pip install 