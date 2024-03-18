r'''Python wrapper of tensorflow ops.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework.load_library import load_op_library as _load
from tensorflow.python.platform import resource_loader as _loader

try:
  _ops = _load(_loader.get_path_to_datafile('libgazer.so'))
except ImportError:
  _ops = None
