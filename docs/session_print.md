# W22: Large Scale Ensembled NLP Systems with Docker and Kubernetes
## AMIA 2019, Washington, DC, 11/17/2019


## 1. Accesse the Virtual Machine (VM)
     ssh amia@<your_ip_address>

The password is (case sensitive): `Tutorial2019`

## 2. Explore your cloud image

Common Linux commands: 

     ls     # list the files in a directory
     pwd    # display the current working directory
     cd     # change directories
     less   # scroll the contents of the file
     cat    # display all contents of file at once
     nano   # a way to edit text
     vim    # another, more complicated way to edit text

The use of `-la` to list current directory with full attributes, and `~/` is equivalent to `/home/amia`. 

To look into the OS release, home directory and `/bin`: 

     cat /etc/os-release
     ls /home/amia
     ls /bin



## 3. Basic Docker Commands (Docker build!)
 1. Now list available docker images on VM: 

        docker images  
        docker --help

 1. Pull a docker image (`whalesay`) from Docker hub:

        docker pull docker/whalesay 

1. Run docker image's version of "Hello World": 

        docker run docker/whalesay cowsay "Hello W22" 
     
## 4. Let's use Docker!
 1. Pull our ML image from the `nlpieumn` repository from Docker hub:

        docker pull nlpieumn/ml 

 2. Start with `bash` in the ML image:

        docker run -t nlpieumn/ml /bin/bash 
     
 3. Explore your docker image:

        cat /etc/os-release
        ls /home/tutorial
        ls /bin 
        exit   # when you're done
     
## 5. Build your own docker image
  
Build the `vote` image from Dockerfile at target "vote":

    cd tutorial
    docker build -t nlpieumn/vote --target vote . 
     
## 6. Let's use Kubernetes! 

Build spec, `kubectl get es/svc`, `kubectl` run dnstools

 1. List all nodes in "cluster":
     
        kubectl get nodes 

 2. List the config file that allows unprivileged user to run commands:

        ls /home/amia/.kube 

 3. List all pods in the default namespace, all namespaces and all services:

        kubectl get pods 
        kubectl get pods --all-namespaces
        kubectl get services
     
## 7. Ensemble models for Word Sense Disambiguation (WSD) 

The standalone script can also run as `python ~/tutorial/scripts/ml.py`. 

To run svm classifier from within docker image:

     docker run -it -e DOCKER='True' -v /home/amia/tutorial:/data nlpieumn/ml 
     /bin/bash -c python /home/tutorial/ml.py -c svm 
     
## 8. Let's use Argo!

 1. List all argo workflows:

        argo list

 2. Validate yaml file:

        nano specs/evaluation.yaml 
        argo lint specs/evaluation.yaml 

 3. Submit argo workflow spec and watch status in real time:

        argo submit --watch specs/evaluation.yaml 

 4. List workflow pods in workflow and view the logs:

        argo get <workflow_name> 
        argo logs <pod_name> 
