# DOCKER CHEAT SHEET 

## Help (one page at a time)

```
docker --help | more
```

NB: Also see [Docker Cheat Sheet](https://www.docker.com/sites/default/files/d8/2019-09/docker-cheat-sheet.pdf) 

## Build image in multi-stage Dockerfile:
Notice the dot at the end (this specifies the directory in which Dockerfile is contained. 
Also, of note: target tells with block of code to use in the multi-stage build Dockerfile. 
```
DOCKER_BUILDKIT=1 docker build -t <image name> --target <target name> .
```

## List all docker images on compute node

```
docker images
```


## Pull image from Docker Hub

```
docker pull docker/whalesay
```

## Pull image from other repository

```
docker pull nlpieumn/ml
```

## Run image:

### Hello World

```
docker run docker/whalesay cowsay boo
```

### ssh into docker container

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/ml /bin/bash
```

#### Explore container

```
pwd
ls -la
ls -la /data
cat /etc/os-release
ls /bin
exit
```

### Example to run help to see available ml classifiers

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/ml /bin/bash -c "python /home/tutorial/ml.py --help"
```

### Example to run default - svm - classifier

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/ml /bin/bash -c "python /home/tutorial/ml.py"
```

### Example to run mlp classifier (or log/rf/bag/boost)

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/ml /bin/bash -c "python /home/tutorial/ml.py -c mlp"
```

### Example to run voting ML ensemble classifier

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/ml /bin/bash -c "python /home/tutorial/ensemble.py"
```

### Example to run cnn/keras classifier

```
docker run -it -v /home/amia/tutorial/:/data nlpieumn/cnn /bin/bash -c "PYTHONHASHSEED=0 python /home/tutorial/cnn.py"
```

### Run docker container using mlp as background daemon process

```
docker run -d -it -v /home/amia/tutorial/:/data nlpieumn/ml /bin/bash -c "python /home/tutorial/ml.py -c mlp"
```

### Stop daemon process container 
(NB: use container id returned from previous step, or get from running “docker ps”)
```
docker stop <conatiner id>
```

### Push to nlpieumn org repo 
(NB: need to have adequate privs to do this!), specify version number `n`

```
docker push nlpieumn/ml
```

### Remove docker image
```
docker rmi <image id> # get <image id> from running “docker images”
```

### Monitor docker processes/resource utilization

```
docker ps
```

```
docker stats
```
