# build image for glove300 embeddings
# to build a particular target, speficy:
# DOCKER_BUILDKIT=1 docker build -t <imagename> --target <target> .

FROM continuumio/miniconda3 AS ml

RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk
RUN pip install gensim
RUN pip install click

RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]

COPY scripts /home/tutorial

WORKDIR /home/tutorial

# build image for cnn  
FROM continuumio/miniconda3 AS cnn

RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk
RUN pip install gensim
RUN pip install click
RUN pip install tensorflow==1.14.0
RUN pip install keras==2.2.4

RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]

COPY scripts /home/tutorial

WORKDIR /home/tutorial

# build image for vote  
FROM continuumio/miniconda3 AS vote

RUN pip install sklearn
RUN pip install pandas

COPY scripts /home/tutorial

WORKDIR /home/tutorial
