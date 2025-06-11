##################################
###  Install env on STAMPEDE3  ###
##################################

conda create --prefix AstroVisBench-env python=3.10 --yes
conda activate AstroVisBench-env

pip install certifi
pip install mpi4py
pip install astropy-iers-data
# pip install importlib  ## This is is not needed in Python 3.4+
grep -Ev '^[[:space:]]*#|^[[:space:]]*$' requirements.txt | xargs -n1 -I{} pip install {} --no-deps
pip install ray
pip install --upgrade ipython ipykernel ipywidgets jupyterlab autoenum
