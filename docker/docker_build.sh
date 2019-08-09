#!/bin/bash
#
# This script runs docker build to create the maskrcnn-gpu docker image.
#

set -exu
sudo nvidia-docker build --tag Fast-SCNN-pytorch-1.0 ./
