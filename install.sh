#!/bin/bash

# Script to set up the TnpB environment

# Step 1: Clone the GitHub repository
echo "Cloning the TnpB repository from GitHub..."
git clone https://github.com/uzh-dqbm-cmi/Tnpb.git

# Navigate into the repository directory
cd Tnpb

# Step 2: Create a Conda environment
echo "Creating a Conda environment named Tnpb..."
conda create --name Tnpb python=3.6 -y

# Step 3: Activate the Conda environment
echo "Activating the Conda environment..."
conda activate Tnpb

# Step 4: Install required packages
echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "Installation completed successfully."

