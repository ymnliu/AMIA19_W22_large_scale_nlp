# build image for glove300 model
FROM continuumio/miniconda3 AS glove300

COPY scripts /home/tutorial
COPY data /home/tutorial/data

RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk
RUN pip install gensim

RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]

WORKDIR /home/tutorial


