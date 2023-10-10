FROM continuumio/miniconda3

WORKDIR /home/DeepLocRNA
 
COPY DeepLocRNA DeepLocRNA
COPY setup.py .
COPY README.md .
COPY webserver/requirements.txt .
RUN conda install python=3.8
RUN pip install -r requirements.txt # server has additional requirements

COPY webserver/run.sh .
RUN chmod +x run.sh
