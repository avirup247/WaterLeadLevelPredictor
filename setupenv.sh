
rm -rf setupenv.sh.*

conda create -n stock-tensorflow python matplotlib ipykernel psutil pandas gitpython

source activate stock-tensorflow

pip install tensorflow==2.3.0 cxxfilt

conda deactivate
/.conda/envs/stock-tensorflow/bin/python -m ipykernel install --user --name=stock-tensorflow
