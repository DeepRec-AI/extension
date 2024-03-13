'''setup script.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


NAME = 'gazer'
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


class NoExtensionBuilder(build_ext):
  r'''Build extensions to do nothing.
  '''
  def build_extension(self, ext):
    return


setup(
    name=NAME,
    version=VERSION,
    packages=PACKAGES,
    include_package_data=True,
    package_data=PACKAGE_DATA,
    install_requires=REQUIRES,
    ext_modules=[Extension('', sources=[])],
    cmdclass={'build_ext': NoExtensionBuilder},
    distclass=BinaryDistribution,
    zip_safe=False,
    author="Aliyun PAI",
    description='Metrics System for DeepLearning.',
    long_description=(
        "Gazer is a metrics system for DeepRec or TenosrFlow."),
    long_description_content_type='text/markdown',
    url="https://github.com/DeepRec-AI",
    download_url='',
    keywords=('deep learning', 'recommendation system'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache License 2.0',
    license_files=('LICENSE', 'NOTICE'),
)
