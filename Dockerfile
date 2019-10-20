# -----------------------------------
FROM continuumio/miniconda3 AS glove300

COPY data /home/tutorial/data
COPY models /home/tutorial/models

#COPY examples  /usr/share/mtap/hello
RUN pip install numpy
RUN pip install sklearn
RUN pip install pandas
RUN pip install nltk

WORKDIR /home/tutorial

ENTRYPOINT [ "python", "svm-glove300.py"]

