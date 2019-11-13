#!/bin/bash

auth=$(cat auth)
tag="amiaW22"

curl -s -X GET -H "Content-Type: application/json" -H "Authorization: Bearer ${auth}" "https://api.digitalocean.com/v2/droplets?tag_name=$tag" | jq -r ".droplets|.[]|.networks|.v4|.[0]|.ip_address"
