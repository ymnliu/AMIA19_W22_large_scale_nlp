#!/bin/bash

userdata=$(cat user-data | jq -sR)
auth=$(cat auth)
tag="amiaW22"

num_drops=$1

for ((num=1;num<=num_drops;num++));
do

    req=$(cat <<EOF
{"name":"amia.W22-${num}",
"region":"nyc3",
"size":"s-2vcpu-4gb",
"image":53969279,
"ssh_keys":[22413657,6230202,6211596],
"backups":false,
"ipv6":false,
"user_data":$userdata,
"private_networking":null,
"volumes": null,
"tags":["$tag"]}
EOF
       )
    droplet=$(curl -sS -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${auth}" -d "$req" "https://api.digitalocean.com/v2/droplets" | jq -r ".droplet.id")
    confirm=$(curl -sS -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${auth}" -d "{\"resources\": [\"do:droplet:$droplet\"]}" "https://api.digitalocean.com/v2/projects/d4a30290-388a-4907-8a35-417feef5bf74/resources")

done

ip_list=$(
    for ip in $(curl -s -X GET -H "Content-Type: application/json" -H "Authorization: Bearer ${auth}" "https://api.digitalocean.com/v2/droplets?tag_name=$tag" | jq -r ".droplets|.[]|.networks|.v4|.[0]|.ip_address")
    do
        echo "$ip ansible_user=amia"
    done
       )

cat <<EOF
[test]
$ip_list
EOF
