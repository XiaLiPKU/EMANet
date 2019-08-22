# !/bin/bash
rm ./models/step_* -rf
rm ./models/final.pth
rm ./models/latest.pth
rm ./logdir/* -rf
rm ./__pycache__ -rf
rm ./lib/__pycache__ -rf
rm ./lib/nn/__pycache__ -rf
rm ./lib/nn/modules/__pycache__ -rf
rm ./lib/nn/parallel/__pycache__ -rf
rm ./lib/utils/__pycache__ -rf
rm ./lib/utils/data/__pycache__ -rf
rm *.py.sw*
rm *.pyc*

