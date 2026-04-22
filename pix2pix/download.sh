#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
TARGET_DIR="${SCRIPT_DIR}/data"

echo "Downloading the pix2pix dataset..."
wget -P "${TARGET_DIR}" http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz 
tar -zxvf "${TARGET_DIR}/facades.tar.gz" -C "${TARGET_DIR}"