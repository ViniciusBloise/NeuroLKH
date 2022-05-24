set -e
cd SRC_swig
swig -python LKH.i
python setup.py build_ext -i
# use `-i` (`--inplace`) and the following commands are not needed any more.
# see https://www.swig.org/Doc1.3/Python.html#Python_nn6

# cd ..
# cp SRC_swig/build/lib.*/_LKH.*.so ./
