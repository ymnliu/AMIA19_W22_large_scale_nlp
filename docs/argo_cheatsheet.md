# ARGO CHEAT SHEET

## Validate yaml template:

```
argo lint <template_name.yaml>
```

## Submit Argo workflow template

(use the switch to `--watch` watch a workflow until it completes)

```
argo submit <template_name.yaml>  #returns the workflow id as xxx-workflow-name-xxx
```

## Delete Argo workflow (and associated pods)

```
argo delete xxx-workflow-name-xxx #use the workflow id returned from the 'submit' command above
```

## List all workflows and their status

```
argo list
```

## List pods in the workflow status 

```
argo get xxx-workflow-name-xxx #use the workflow id returned from the 'submit' command above
```

## View pod runtime log

```
argo logs xxx-pod-name-xxx #from 'get' command above
```
