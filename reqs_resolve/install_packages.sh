pip install astropy-iers-data
pip install certifi
grep -Ev '^[[:space:]]*#|^[[:space:]]*$' requirements.txt | xargs -n1 -I{} pip install {} --no-deps
