# Cheatsheet

### Hands On

#### 1. Accessing the Virtual Machine (VM)
     ssh amia@<ip_address>
     Password: Tutorial2019

#### 2. Common Commands (notes: use of `-la`, `~/`, `/home/amia` and possible run of `python ~/tutorial/scripts/ml.py`)
     ls     # list the files in a directory
     pwd    # display the current working directory
     cd     # change directories
     less   # scroll the contents of the file
     cat    # display all contents of file at once
     nano   # a way to edit text
     vim    # another, more complicated way to edit text
     
#### 3. Explore your cloud image(notes: use of `-la`, and `~/`, `/home/amia`)
     cat /etc/os-release
     ls /home/opensuse
     ls /bin
     
#### 4. Basic Docker Commands (notes: Docker build!)
     docker images  
     docker --help
     docker pull docker/whalesay
     docker images
     docker run docker/whalesay cowsay boo
     
#### 5. Let's use Docker!
     docker pull nlpieumn/ml
     docker run -t nlpieumn/ml /bin/bash
     
#### 6. Explore your docker image
     cat /etc/os-release
     ls /home/tutorial
     ls /bin 
     exit   # when you're done
     
#### 7. Let's use Kubernetes! (notes: build spec, `kubectl get es/svc`, `kubectl` run dnstools)
     kubectl get nodes
     ls /home/amia/.kube
     kubectl get pods
     kubectl get pods --all-namespaces
     kubectl get services
     
#### 8. Word Sense Disambiguation (WSD) (run of `python ~/tutorial/scripts/ml.py` from command line)
     docker run -it -e DOCKER='True' -v /home/amia/tutorial:/data nlpieumn/ml 
     /bin/bash -c python /home/tutorial/ml.py -c svm
     
#### 9. Let's use Argo!
     argo list
     nano evaluation.yaml
     argo lint evaluation.yaml
     argo submit evaluation.yaml
     argo watch <workflow_name>
     argo get <workflow_name>
     argo logs <pod_name>
