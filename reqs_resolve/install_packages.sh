pip install astropy-iers-data
pip install certifi
pip install mpi4py
pip install importlib
grep -Ev '^[[:space:]]*#|^[[:space:]]*$' requirements.txt | xargs -n1 -I{} pip install {} --no-deps
