#!/bin/bash
# nvidia_install.sh

# Check if NVIDIA is present
lspci | grep -i nvidia

# Add the NVIDIA repository and update the package list
sudo add-apt-repository ppa:graphics-drivers/ppa --yes
sudo apt update

# Install the NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-535 nvidia-dkms-535 -y
sudo apt install nvidia-cuda-toolkit -y

# Check the status of the drivers
dkms status

# Reboot the system to apply changes
sudo reboot
