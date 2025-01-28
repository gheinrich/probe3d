#!/bin/bash

SUFF="jan25-ftup"
TAG="gitlab-master.nvidia.com/dler/evfm/probe3d:${SUFF}"

docker tag dler/evfm:$SUFF $TAG
docker push $TAG
