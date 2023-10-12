#FROM continuumio/miniconda3
FROM nvidia/cuda:11.0-base

# Install Conda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    /opt/conda/bin/conda install -y python=3.8

WORKDIR /home/DeepLocRNA

COPY DeepLocRNA DeepLocRNA
COPY setup.py .
COPY README.md .
COPY requirements.txt .
#RUN conda install python=3.8
RUN pip install -r requirements.txt # server has additional requirements

COPY webserver/run.sh .
RUN mkdir /outputs
RUN chmod +x run.sh
