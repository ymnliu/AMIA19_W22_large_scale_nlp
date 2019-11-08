# KUBERNETES CHEAT SHEET 


## Help

```
kubectl --help
```

NB: Everything you need to know about `kubectl` usage is found here [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

## Display cluster nodes
```
kubectl get nodes 
kubectl get nodes -o=wide # show IP address, etc.
```

## View kubectl config (allows non-root management of cluster)

```
ls -la /home/opensuse/.kube
```

## Display pods

```
kubectl get pods # display default namespace only
kubectl get pods --all-namespaces # display all namespaces
kubectl get pods --namespace=kube-system # display kube-system namespace
kubectl get pods dnstools #display dnstools pod only from default namespace
```

## Display services and endpoints

```
kubectl get service # display default namespace only -> svc is shorthand for service
kubectl get service --all-namespaces # display all namespaces
kubectl get service --namespace=kube-system # display kube-system namespace

kubectl get ep # display default namespace only 
kubectl get ep --all-namespaces # display all namespaces
kubectl get ep --namespace=kube-system # display kube-system namespace

```






