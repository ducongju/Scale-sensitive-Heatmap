#!/bin/bash
#SBATCH -J sacle-sensitive
#SBATCH -p gpu
#SBATCH -N 1

eval "$(/opt/app/conda/bin/conda shell.bash hook)"
module load app/cuda/10.1

# Install dependencies
conda create -n sacle-sensitive python=3.6
conda activate scale-sensitive
pip install -r requirements.txt

# Install COCOAPI
git clone https://github.com/cocodataset/cocoapi.git
cd "cocoapi/PythonAPI"
python3 setup.py install --user

# Install CrowdPoseAPI
git clone https://github.com/Jeff-sjtu/CrowdPose.git
cd "../../crowdpose-api/PythonAPI"
sh install.sh

# Build dcn model:
cd "../.."
python setup.py develop