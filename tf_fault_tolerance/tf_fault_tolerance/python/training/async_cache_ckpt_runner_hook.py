# Copyright 2023 The DeepRec Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import threading
import weakref

from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook

class AsyncCacheCKPTRunner(object):
  """The runner for asynchronous cache checkpoints."""
  def __init__(self, fetch_op, cancel_op, resume_op):
    """Create a AsyncCacheCKPTRunner.

    When you later call the `create_threads()` method, the
    `AsyncCacheCKPTRunner` will create thread for `fetch_op`.

    Args:
      fetch_op: Op that repeats by this runner.
      cancel_op: Op that stops fetch_op on stop of this runner.
      resume_op: Op that restarts fetch_op on start of this runner.
    """
    try:
      executing_eagerly = context.executing_eagerly()
    except: # pylint: disable=bare-except
      executing_eagerly = context.in_eager_mode()
    else:
      executing_eagerly = False
    if not executing_eagerly:
      self._name = ops.get_default_graph().unique_name(self.__class__.__name__)
    else:
      self._name = context.context().scope_name
    self._fetch_op = fetch_op
    self._cancel_op = cancel_op
    self._resume_op = resume_op
    self._runs_per_session = weakref.WeakKeyDictionary()
    self._exceptions_raised = []

  def _run(self, sess, coord):
    """Run caching checkpoint in thread.
    Args:
      sess: A `Session`.
      coord: A `Coordinator` object for reporting errors and checking stop
        conditions.
    """
    try:
      sess.run(self._resume_op)
      run_fetch = sess.make_callable(self._fetch_op)
      while True:
        try:
          if coord and coord.should_stop():
            break
          run_fetch()
          logging.info("async caching checkpoint was success.")
        except errors.CancelledError:
          logging.info("async caching checkpoint was cancelled.")
          return
    except Exception as e:
      if coord:
        coord.request_stop(e)
        if not isinstance(e, errors.CancelledError):
          logging.error(
            "async caching checkpoint was cancelled unexpectedly:\n\n%s", e)
          raise
      else:
        self._exceptions_raised.append(e)
    finally:
      self._runs_per_session[sess] = False

  def _cancel_on_stop(self, sess, coord):
    """Clean up resources on stop.

    Args:
      sess: A `Session`.
      coord: A `Coordinator` object for reporting errors and checking stop
      conditions.
    """
    coord.wait_for_stop()
    try:
      cancel = sess.make_callable(self._cancel_op)
      cancel()
    except Exception:
      pass


  @property
  def name(self):
    """Name of this Runner"""
    return self._name

  @property
  def exceptions_raised(self):
    """Exceptions raised but not handled by the `AsyncCacheCKPTRunner` thread.

    Exceptions raised in `AsyncCacheCKPTRunner` thread are handled in one of
    two ways depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `AsyncCacheCKPTRunner`.
    * Without a `Coordinator`, exceptions are captured by the
      `AsyncCacheCKPTRunner` and made available in this `exceptions_raised`
      property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
    return self._exceptions_raised

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Create threads to cache checkpoints.

    This method requires a session in which the graph was launched, it creates
    a list of threads, optionally starting them.

    The `coord` argument is an optional coordinator that the threads will use
    to terminate together and report exceptions. If a coordinator is given,
    this method starts an additional thread to cancel when the coordinator
    requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: (Optional.) `Coordinator` object for reporting errors and checking
        stop conditions.
      deamon: (Optional.) Boolean. If `True` make the threads daemon thread.
      start: (Optional.) Boolean. If `True` starts the threads. If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
    try:
      if self._runs_per_session[sess] == True:
        # Already started: no new thread to return.
        return None
    except KeyError:
      pass
    self._runs_per_session[sess] = True

    ret_threads = []
    thread_name = "AsyncCacheCKPTThread-" + self.name
    ret_threads.append(threading.Thread(
        target=self._run, args=(sess, coord), name=thread_name))
    if coord:
      thread_name = "CancelOnStopThread-" + self.name
      ret_threads.append(threading.Thread(
          target=self._cancel_on_stop, args=(sess, coord), name=thread_name))
    for t in ret_threads:
      if coord:
        coord.register_thread(t)
      if daemon:
        t.daemon = True
      if start:
        t.start()

    return ret_threads

class AsyncCacheCKPTRunnerHook(session_run_hook.SessionRunHook):
  """AsyncCacheCKPTRunnerHook that creates AsyncCacheCKPTRunner."""
  def __init__(self, daemon=True, start=True):
    """Build AsyncCacheCKPTRunnerHook.

    Args:
      daemon: (Optional.) Whether the threads should be marked as `daemons`,
        meaning they don't block program exit.
      start: (Optional.) If `False` threads would not be started.
    """
    super(AsyncCacheCKPTRunnerHook, self).__init__()
    self._daemon = daemon
    self._start = start
    self._fetch_op = None
    self._cancel_op = None
    self._resume_op = None
    self._runner = None

  def begin(self):
    with ops.name_scope("save/AsyncSaveCacheCKPT"):
      self._fetch_op = control_flow_ops.no_op(name="async_fetch_op")
      self._cancel_op = control_flow_ops.no_op(name="async_cancel_op")
      self._resume_op = control_flow_ops.no_op(name="async_resume_op")

    self._runner = \
        AsyncCacheCKPTRunner(self._fetch_op, self._cancel_op, self._resume_op)

  def after_create_session(self, session, coord):
    self._runner.create_threads(session, coord, daemon=self._daemon, \
                                start=self._start)
