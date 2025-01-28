#!/bin/bash

docker build -t dler/probe3d:jan25-ftup --progress plain $@ -f Dockerfile .
