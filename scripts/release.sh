cd ..
rm -rf dist
uv build --sdist
# Before uploading with twine, fetch the wheels from github action arfifact and
# put them inside the 'dist' folder
# uv publish dist/*
