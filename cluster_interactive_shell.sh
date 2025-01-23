#!/bin/bash

PARTITION=${1:-"interactive"}
GPUS=${2:-2}
PROJECT=${3:-"."}
RESERVATION=${4:-""}
# GPUS=${2:-8}

# On aws, the correct drives are auto-mounted
MOUNT_CMD=""
IMAGE_SUFFIX=""
DURATION="4"
CONSTRAINT=""
# if [[ $HOSTNAME != "draco-aws-login-01" ]]; then
#     MOUNT_CMD="--mounts $MOUNTS"
#     CONSTRAINT="--constraint 'gpu_32gb'"
# else
#     IMAGE_SUFFIX="-aws"
#     DURATION="4"
# fi

IMAGE=`cat $PROJECT/docker_image`
if [[ ! -f $IMAGE ]]; then
    IMAGE="${IMAGE}${IMAGE_SUFFIX}"
fi

WORKDIR=`pwd`
if [[ $WORKDIR != "." ]]; then
    WORKDIR="${WORKDIR}/${PROJECT}"
fi

if [[ $HOSTNAME == "draco-oci-login" || $HOSTNAME == "cs-oci-ord-login-01" ]]; then
    ADDL="--more_srun_args=--gpus-per-node=$GPUS"
fi

if [[ $PARTITION == "batch" ]]; then
    DURATION="4"
fi

if [[ $HOSTNAME == "cs-oci-ord-login-01" && $PARTITION == "interactive" ]]; then
    DURATION="2"
fi

if [[ -n "$RESERVATION" ]]; then
    RESERVATION="--reservation $RESERVATION"
fi

submit_job \
           --gpu $GPUS \
           --partition "$PARTITION" \
           $CONSTRAINT \
           $MOUNT_CMD \
           --workdir $WORKDIR \
           --image $IMAGE \
           --coolname \
           --interactive \
           --duration $DURATION \
           $RESERVATION \
           $ADDL \
           -c "bash"
