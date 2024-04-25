'''setup script.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from pybind11.setup_helpers import Pybind11Extension, build_ext

ROOT_PATH = os.path.dirname(os.path.abspath(os.path.join(os.getcwd())))

include_dirs=[]
library_dirs=[]
libraries = []
include_dirs.append(ROOT_PATH)
include_dirs.append(ROOT_PATH + '/deeprec_master/include')
include_dirs.append(ROOT_PATH + '/third_party/pybind11/pybind11/include')
include_dirs.append(ROOT_PATH + '/third_party/grpc/build/include')
include_dirs.append(ROOT_PATH + '/third_party/protobuf/build/include')
include_dirs.append(numpy.get_include())

libraries.append('deeprec_master')
library_dirs.append(ROOT_PATH + '/deeprec_master/deeprec_master')

extra_link_args=[]
extra_compile_args = []
extra_link_args.append('-Wl,-rpath=$ORIGIN/')
extra_compile_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
extra_compile_args.append('-std=gnu++11')
ext_modules = [
    Extension(
        "deeprec_master.pywrap_deeprecmaster",
        ["deeprec_master/python/py_export.cc"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    ),
]

NAME = 'deeprec_master'
VERSION = '1.0'
PACKAGES = find_packages(exclude=['cc'])
PACKAGE_DATA = {'': ['*.so', '*.so.*']}
REQUIRES = os.getenv('WHEEL_REQUIRES', '').split(';')


class BinaryDistribution(Distribution):
  r'''This class is needed in order to create OS specific wheels.
  '''
  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


setup(
    name=NAME,
    version=VERSION,
    packages=PACKAGES,
    include_package_data=True,
    package_data=PACKAGE_DATA,
    install_requires=REQUIRES,
    ext_modules=ext_modules,
    distclass=BinaryDistribution,
    zip_safe=False,
    author="Aliyun PAI",
    description='Scaling Controller for DeepRec.',
    long_description=(
        "DeepRec Master is the controller for DeepRec."),
    long_description_content_type='text/markdown',
    url="",
    download_url='',
    keywords=('deep learning', 'recommendation system'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache License 2.0',
    license_files=('LICENSE', 'NOTICE'),
)

