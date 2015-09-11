from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

gibbs_inner = Extension('gibbs_inner',
                        sources=['cptm/gibbs_inner.pyx'])

setup(name='cptm',
      packages=['cptm'],
      install_requires=['cython'],
      ext_modules=[gibbs_inner],
      cmdclass={'build_ext': build_ext},
      include_dirs=[np.get_include()])
