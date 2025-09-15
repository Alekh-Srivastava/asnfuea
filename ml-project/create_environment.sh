#!/bin/bash

# Check if Anaconda is installed
if command -v conda &> /dev/null; then
    echo "Anaconda is already installed."
else
    echo "Anaconda is not installed. Installing now..."
    # Download and install Anaconda
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O ~/anaconda.sh
    bash ~/anaconda.sh -b -p $HOME/anaconda
    
    # Add Anaconda to PATH
    echo 'export PATH="$HOME/anaconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
    echo "Anaconda installation complete."
fi

# Create a new Conda environment
echo "Creating a new Conda environment named ml-project..."
conda create -n ml-project python=3.9 -y

# Activate the environment
echo "Activating the ml-project environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml-project

# Install dependencies
echo "Installing dependencies from dev_requirements.txt..."
pip install -r dev_requirements.txt

# Install Jupyter kernel
echo "Setting up Jupyter kernel for ml-project environment..."
python -m ipykernel install --user --name ml-project --display-name "Python (ml-project)"

echo "Environment setup complete. You can now activate the environment with: conda activate ml-project"
