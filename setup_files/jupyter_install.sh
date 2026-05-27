#!/bin/bash
# jupyter_install.sh

# Update and install pip3
sudo apt update
sudo apt install python3-pip -y

# Install Jupyter Notebook
pip3 install jupyter notebook --user

# Install Jupyter container
sudo apt install jupyter-core -y

# Launch Jupyter Notebook with access from all IPs
python3 -m notebook --ip 0.0.0.0
