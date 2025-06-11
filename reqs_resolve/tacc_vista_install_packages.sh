##############################
###  Install env on VISTA  ###
##############################

conda create --prefix AstroVisBench-env python=3.10 --yes
conda activate AstroVisBench-env

## Vista cluster needs newer GCC and CUDA versions
module load gcc/13 cuda/12.4 nvidia_math/12.4 nccl/12.4
export CC=gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST="9.0 9.0a"

pip install certifi
pip install mpi4py
pip install astropy-iers-data
# pip install importlib  ## This is is not needed in Python 3.4+
grep -Ev '^[[:space:]]*#|^[[:space:]]*$' requirements.txt | xargs -n1 -I{} pip install {} --no-deps
pip install ray
pip install --upgrade ipython ipykernel ipywidgets jupyterlab autoenum
