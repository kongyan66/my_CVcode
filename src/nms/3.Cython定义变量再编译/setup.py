from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'nms_module',
    ext_modules = cythonize('nms_np_c.pyx'),
    include_dirs = [numpy.get_include()]
)