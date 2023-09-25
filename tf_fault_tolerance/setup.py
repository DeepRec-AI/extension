# Copyright 2023 The DeepRec Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


NAME = 'tf-fault-tolerance'
VERSION = '1.0'
PACKAGES = find_packages(exclude=['cc'])
PACKAGE_DATA = {'': ['*.so', '*.so.*']}
REQUIRES = [
    'tensorflow <= 1.15.5',
]

class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False

class NoExtensionBuilder(build_ext):
  r'''Build extensions to do nothing.
  '''
  def build_extension(self, ext):
    return

print("aaaa {}".format(PACKAGES))

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
    author='Aliyun PAI.',
    description=('tf-fault-tolerance is a fault-tolerant component for distributed tensorflow training tasks'),
    url="",
    download_url='',
    keywords=('machine learning', 'tensorflow', 'fault tolerance'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    license_files=('LICENSE', 'NOTICE'),
)
