# DOCKER CHEAT SHEET 


## Build image in multi-stage Dockerfile:
notice the dot at the end (this specifies the directory in which Dockerfile is contained. 
Also, of note: target tells with block of code to use in the multi-stage build Dockerfile. 
```
DOCKER_BUILDKIT=1 docker build -t <image name> --target <target name> .
```

## Run image:

### ssh into docker container

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/glove300:2 /bin/bash
```

### example to run default - svm - classifier

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/glove300 /bin/bash -c "python /home/tutorial/glove300.py"
```

### example to run mlp classifier

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/glove300 /bin/bash -c "python /home/tutorial/glove300.py -c mlp"
```
### example to run cnn/keras classifier

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/glove300 /bin/bash -c "PYTHONHASHSEED=0 python /home/tutorial/cnn.py"
```

### example to run help

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/glove300 /bin/bash -c "python /home/tutorial/glove300.py --help"
```

### run docker container using mlp as background daemon process

```
docker run -d -it -v /home/amia/tutorial/:/data nlpieumn/glove300 /bin/bash -c "python /home/tutorial/glove300.py -c mlp"
```

### stop daemon process container 
(NB: use container id returned from previous step, or get from running “docker ps”)
```
docker stop <conatiner id>
```

### push to nlpieumn org repo 
(NB: need to have adequate privs to do this!), specify version number `n`

```
docker push nlpieumn/glove300
```

### pull from repo
```
docker pull nlpieumn/glove300
```

### list all docker images on compute node

```
docker images
```

### remove docker image
```
docker rmi <image id> # get <image id> from running “docker images”
```

### monitor docker resource utilization
```
docker stats
```
