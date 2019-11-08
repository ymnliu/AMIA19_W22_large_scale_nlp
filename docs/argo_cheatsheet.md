# ARGO CHEAT SHEET

## Help 

```
argo --help
```

NB: Also see [Argo Getting Started](https://github.com/argoproj/argo/blob/master/demo.md) 

## Validate yaml template:

```
argo lint <template_name.yaml> # e.g., argo lint ~/tutorial/evaluation.yaml
```

## Submit Argo workflow template

(use the switch to `--watch` watch a workflow until it completes)

```
argo submit <template_name.yaml>  # returns the workflow id as xxx-workflow-name-xxx
# e.g., submit lint ~/tutorial/complete_evaluation.yaml
```

## List all workflows and their status

```
argo list
```

## Delete Argo workflow (and associated pods)

```
argo delete xxx-workflow-name-xxx #use the workflow id returned from the 'submit' command above
```


## List pods in the workflow status 

```
argo get xxx-workflow-name-xxx #use the workflow id returned from the 'submit' command above
```

## View pod runtime log

```
argo logs xxx-pod-name-xxx #from 'get' command above
kubectl logs xxx-pod-name-xxx -c main # query main container of pod for log
```
