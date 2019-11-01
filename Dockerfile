# build image for glove300 embeddings
FROM continuumio/miniconda3 AS glove300

COPY scripts /home/tutorial

RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk
RUN pip install gensim
RUN pip install click

RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]

WORKDIR /home/tutorial

# build image for cnn  
FROM continuumio/miniconda3 AS cnn

COPY scripts /home/tutorial
COPY data /home/tutorial/data

RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk
RUN pip install gensim
RUN pip install click
RUN pip install tensorflow==1.14.0
RUN pip install keras==2.2.4

RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]

WORKDIR /home/tutorial

# build image for ensembling  
FROM continuumio/miniconda3 AS ensemble

COPY scripts /home/tutorial

RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk
RUN pip install gensim

RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]

WORKDIR /home/tutorial
