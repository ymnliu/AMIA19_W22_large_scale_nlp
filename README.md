# W22: Large Scale Ensembled NLP Systems with Docker and Kubernetes

https://www.amia.org/amia2019/workshops


## Introduction
This workshop will provide attendees the information necessary to implement NLP workflows using cloud native technologies by providing practical introductions to UIMA-AS, Docker, Kubernetes, and Argo. It will start with the basics of composing NLP system "ensembles" designed to optimize performance in a particular domain and proceed through an introduction to cloud technologies-- including core concepts and technical terms, and explanation of several alternatives to the Argo/Kubernetes/Docker workflow. Explanations of when, where, and why to use each technology, along with some of the practical challenges of using each in a high-security (PHI) environment will be discussed. Workshop participants will then install Docker (a container protocol and server), Kubernetes (a container orchestration system), minikube (a platform for using Kubernetes locally), and Argo (a Kubernetes workflow manager) on their own computers and run a test NLP workflow on a collection of exemplar clinical notes (from the MTSamples corpus). We will then discuss common architectures for UIMA pipelines and pipelines for technologies that are common in other informatics domains and non-UIMA tools, as time permits.

## Repository Structure

- `data`: a subset of UMN WSD dataset for demo
- `docs`: documentation of argo, docker and k8s workflow 
- `models`: word to index dictionary and Glove truncated word embedding models 
- `scripts`: Python scripts of WSD methods used in the tutorial
- `specs`: yaml config files for argo/k8s.


## Contributors

- Raymond Finzel, Greg Silverman, Shreya Datar,  Serguei Pakhomov, University of Minnesota; 
- Sijia Liu, Hongfang Liu, Mayo Clinic; 
- Xiaoqian Jiang, UTHealth;
