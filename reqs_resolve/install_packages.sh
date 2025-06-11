pip install certifi
pip install mpi4py
pip install astropy-iers-data
# pip install importlib  ## This is is not needed in Python 3.4+
grep -Ev '^[[:space:]]*#|^[[:space:]]*$' requirements.txt | xargs -n1 -I{} pip install {} --no-deps