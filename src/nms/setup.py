# 编译.pyx文件 生成.so文件，可以直接导入使用
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'nms_module',
    ext_modules = cythonize('nms_np_c.pyx')
)

# python setup.py build