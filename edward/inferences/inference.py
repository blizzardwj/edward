# TODO: inference.py is a general collection of utility functions that
# each specific file shares; it's less structured and more of a bag.
# + maybe move to util/inference.py?
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf
import os

from datetime import datetime
from edward.models import RandomVariable
from edward.util import get_session, get_variables, Progbar
from edward.util import transform as _transform

from tensorflow.contrib.distributions import bijectors


def process_inputs(latent_vars=None, data=None):
  """
  Args:
    latent_vars: dict, optional.
      Collection of latent variables (of type `RandomVariable` or
      `tf.Tensor`) to perform inference on. Each random variable is
      binded to another random variable; the latter will infer the
      former conditional on data.
    data: dict, optional.
      Data dictionary which binds observed variables (of type
      `RandomVariable` or `tf.Tensor`) to their realizations (of
      type `tf.Tensor`). It can also bind placeholders (of type
      `tf.Tensor`) used in the model to their realizations; and
      prior latent variables (of type `RandomVariable`) to posterior
      latent variables (of type `RandomVariable`).
  """
  sess = get_session()
  if latent_vars is None:
    latent_vars = {}
  if data is None:
    data = {}

  check_latent_vars(latent_vars)

  check_data(data)
  processed_data = {}
  for key, value in six.iteritems(data):
    if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
      processed_data[key] = value
    elif isinstance(key, (RandomVariable, tf.Tensor)):
      if isinstance(value, (RandomVariable, tf.Tensor)):
        processed_data[key] = value
      elif isinstance(value, (float, list, int, np.ndarray, np.number, str)):
        # If value is a Python type, store it in the graph.
        # Assign its placeholder with the key's data type.
        with tf.variable_scope(None, default_name="data"):
          ph = tf.placeholder(key.dtype, np.shape(value))
          var = tf.Variable(ph, trainable=False, collections=[])
          sess.run(var.initializer, {ph: value})
          processed_data[key] = var
  return latent_vars, processed_data


def transform(latent_vars, auto_transform=True):
  """
  Args:
    auto_transform: bool, optional.
      Whether to automatically transform continuous latent variables
      of unequal support to be on the unconstrained space. It is
      only applied if the argument is `True`, the latent variable
      pair are `ed.RandomVariable`s with the `support` attribute,
      the supports are both continuous and unequal.
  """
  # map from original latent vars to unconstrained versions
  if auto_transform:
    latent_vars_temp = latent_vars.copy()
    # latent_vars maps original latent vars to constrained Q's.
    # latent_vars_unconstrained maps unconstrained vars to unconstrained Q's.
    latent_vars = {}
    latent_vars_unconstrained = {}
    for z, qz in six.iteritems(latent_vars_temp):
      if hasattr(z, 'support') and hasattr(qz, 'support') and \
            z.support != qz.support and qz.support != 'point':

        # transform z to an unconstrained space
        z_unconstrained = _transform(z)

        # make sure we also have a qz that covers the unconstrained space
        if qz.support == "points":
          qz_unconstrained = qz
        else:
          qz_unconstrained = _transform(qz)
        latent_vars_unconstrained[z_unconstrained] = qz_unconstrained

        # additionally construct the transformation of qz
        # back into the original constrained space
        if z_unconstrained != z:
          qz_constrained = _transform(
            qz_unconstrained, bijectors.Invert(z_unconstrained.bijector))

          try: # attempt to pushforward the params of Empirical distributions
            qz_constrained.params = z_unconstrained.bijector.inverse(
              qz_unconstrained.params)
          except: # qz_unconstrained is not an Empirical distribution
            pass

        else:
          qz_constrained = qz_unconstrained

        latent_vars[z] = qz_constrained
      else:
        latent_vars[z] = qz
        latent_vars_unconstrained[z] = qz
    del latent_vars_temp
  else:
    latent_vars_unconstrained = None
  return latent_vars, latent_vars_unconstrained


def process_dict(x):
  if x is None:
    x = {}
  elif not isinstance(x, dict):
    raise TypeError("x must be dict; got {}".format(type(x).__name__))
  return x


def summary_key():
  return tf.get_default_graph().unique_name("summaries")


def set_log_variables(summary_key, latent_vars=None, data=None, log_vars=None):
  """Log variables to TensorBoard.

  For each variable in `log_vars`, forms a `tf.summary.scalar` if
  the variable has scalar shape; otherwise forms a `tf.summary.histogram`.

  Args:
    log_vars: list, optional.
      Specifies the list of variables to log after each `n_print`
      steps. If None, will log all variables. If `[]`, no variables
      will be logged.
  """
  if log_vars is None:
    log_vars = []
    for key in six.iterkeys(data):
      log_vars += get_variables(key)

    for key, value in six.iteritems(latent_vars):
      log_vars += get_variables(key)
      log_vars += get_variables(value)

    log_vars = set(log_vars)

  for var in log_vars:
    # replace colons which are an invalid character
    var_name = var.name.replace(':', '/')
    # Log all scalars.
    if len(var.shape) == 0:
      tf.summary.scalar("parameter/{}".format(var_name),
                        var, collections=[summary_key])
    elif len(var.shape) == 1 and var.shape[0] == 1:
      tf.summary.scalar("parameter/{}".format(var_name),
                        var[0], collections=[summary_key])
    else:
      # If var is multi-dimensional, log a histogram of its values.
      tf.summary.histogram("parameter/{}".format(var_name),
                           var, collections=[summary_key])


def run(train_op, summary_key=None, n_iter=1000, n_print=None,
        logdir=None, log_timestamp=True, log_vars=None,
        debug=False, variables=None,
        use_coordinator=True, *args, **kwargs):
  """A simple wrapper to run inference.

  1. Initialize algorithm via `initialize`.
  2. (Optional) Build a TensorFlow summary writer for TensorBoard.
  3. (Optional) Initialize TensorFlow variables.
  4. (Optional) Start queue runners.
  5. Run `update` for `n_iter` iterations.
  6. While running, `print_progress`.
  7. Finalize algorithm via `finalize`.
  8. (Optional) Stop queue runners.

  To customize the way inference is run, run these steps
  individually.

  Args:
    n_iter: int, optional.
      Number of iterations for algorithm when calling `run()`.
      Alternatively if controlling inference manually, it is the
      expected number of calls to `update()`; this number determines
      tracking information during the print progress.
    n_print: int, optional.
      Number of iterations for each print progress. To suppress print
      progress, then specify 0. Default is `int(n_iter / 100)`.
    logdir: str, optional.
      Directory where event file will be written. For details,
      see `tf.summary.FileWriter`. Default is to log nothing.
    log_timestamp: bool, optional.
      If True (and `logdir` is specified), create a subdirectory of
      `logdir` to save the specific run results. The subdirectory's
      name is the current UTC timestamp with format 'YYYYMMDD_HHMMSS'.
    log_vars: list, optional.
      Specifies the list of variables to log after each `n_print`
      steps. If None, will log all variables. If `[]`, no variables
      will be logged. `logdir` must be specified for variables to be
      logged.
    debug: bool, optional.
      If True, add checks for `NaN` and `Inf` to all computations
      in the graph. May result in substantially slower execution
      times.
    variables: list, optional.
      A list of TensorFlow variables to initialize during inference.
      Default is to initialize all variables (this includes
      reinitializing variables that were already initialized). To
      avoid initializing any variables, pass in an empty list.
    use_coordinator: bool, optional.
      Whether to start and stop queue runners during inference using a
      TensorFlow coordinator. For example, queue runners are necessary
      for batch training with file readers.
  """
  if n_print is None:
    n_print = int(n_iter / 100)
  progbar = Progbar(n_iter)
  t = tf.Variable(0, trainable=False, name="iteration")
  kwargs['t'] = t.assign_add(1)  # add to update()

  if summary_key is not None:
    summarize = tf.summary.merge_all(key=summary_key)
    if log_timestamp:
      logdir = os.path.expanduser(logdir)
      logdir = os.path.join(
          logdir, datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S"))
    train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
  else:
    summarize = None
    train_writer = None

  if debug:
    op_check = tf.add_check_numerics_ops()
  else:
    op_check = None

  if variables is None:
    init = tf.global_variables_initializer()
  else:
    init = tf.variables_initializer(variables)

  # Feed placeholders in case initialization depends on them.
  feed_dict = kwargs.get('feed_dict', {})
  init.run(feed_dict)

  if use_coordinator:
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

  for _ in range(n_iter):
    info_dict = update(train_op, n_print, summarize, train_writer,
                       debug, op_check, *args, **kwargs)
    print_progress(progbar, n_print, info_dict)

  finalize = None
  if finalize is not None:
    finalize_ops = finalize()
    sess = get_session()
    sess.run(finalize_op, feed_dict)
  else:
    if summary_key is not None:
      train_writer.close()

  if use_coordinator:
    # Ask threads to stop.
    coord.request_stop()
    coord.join(threads)

  # TODO should go before run but after it's created
  # TODO need scope
  # To reset inference (e.g., internal variable counters incremented
  # over training), fetch inference's reset ops from session with
  # `sess.run(inference.reset)`.
  #
  # Store reset ops which user can call. Subclasses should append
  # any ops needed to reset internal variables in inference.
  # reset = [tf.variables_initializer([t])]
  # reset.append(tf.variables_initializer(
  #     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)))


def check_data(data):
  """Check that the data dictionary passed during inference and
  criticism is valid.
  """
  if not isinstance(data, dict):
    raise TypeError("data must have type dict.")

  for key, value in six.iteritems(data):
    if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
      if isinstance(value, RandomVariable):
        raise TypeError("The value of a feed cannot be a ed.RandomVariable "
                        "object. "
                        "Acceptable feed values include Python scalars, "
                        "strings, lists, numpy ndarrays, or TensorHandles.")
      elif isinstance(value, tf.Tensor):
        raise TypeError("The value of a feed cannot be a tf.Tensor object. "
                        "Acceptable feed values include Python scalars, "
                        "strings, lists, numpy ndarrays, or TensorHandles.")
    elif isinstance(key, (RandomVariable, tf.Tensor)):
      if isinstance(value, (RandomVariable, tf.Tensor)):
        if not key.shape.is_compatible_with(value.shape):
          raise TypeError("Key-value pair in data does not have same "
                          "shape: {}, {}".format(key.shape, value.shape))
        elif key.dtype != value.dtype:
          raise TypeError("Key-value pair in data does not have same "
                          "dtype: {}, {}".format(key.dtype, value.dtype))
      elif isinstance(value, (float, list, int, np.ndarray, np.number, str)):
        if not key.shape.is_compatible_with(np.shape(value)):
          raise TypeError("Key-value pair in data does not have same "
                          "shape: {}, {}".format(key.shape, np.shape(value)))
        elif isinstance(value, (np.ndarray, np.number)) and \
                not np.issubdtype(value.dtype, np.float) and \
                not np.issubdtype(value.dtype, np.int) and \
                not np.issubdtype(value.dtype, np.str):
          raise TypeError("Data value has an invalid dtype: "
                          "{}".format(value.dtype))
      else:
        raise TypeError("Data value has an invalid type: "
                        "{}".format(type(value)))
    else:
      raise TypeError("Data key has an invalid type: {}".format(type(key)))


def check_latent_vars(latent_vars):
  """Check that the latent variable dictionary passed during inference and
  criticism is valid.
  """
  if not isinstance(latent_vars, dict):
    raise TypeError("latent_vars must have type dict.")

  for key, value in six.iteritems(latent_vars):
    if not isinstance(key, (RandomVariable, tf.Tensor)):
      raise TypeError("Latent variable key has an invalid type: "
                      "{}".format(type(key)))
    elif not isinstance(value, (RandomVariable, tf.Tensor)):
      raise TypeError("Latent variable value has an invalid type: "
                      "{}".format(type(value)))
    elif not key.shape.is_compatible_with(value.shape):
      raise TypeError("Key-value pair in latent_vars does not have same "
                      "shape: {}, {}".format(key.shape, value.shape))
    elif key.dtype != value.dtype:
      raise TypeError("Key-value pair in latent_vars does not have same "
                      "dtype: {}, {}".format(key.dtype, value.dtype))

## from variational_inference.py

def process_var_list(var_list, latent_vars, data):
  """
  Returns:
    List of TensorFlow variables to optimize over. Default is all
    trainable variables that `latent_vars` and `data` depend on,
    excluding those that are only used in conditionals in `data`.
  """
  # Traverse random variable graphs to get default list of variables.
  if var_list is None:
    var_list = set()
    trainables = tf.trainable_variables()
    for z, qz in six.iteritems(latent_vars):
      var_list.update(get_variables(z, collection=trainables))
      var_list.update(get_variables(qz, collection=trainables))

    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable) and \
              not isinstance(qx, RandomVariable):
        var_list.update(get_variables(x, collection=trainables))

    var_list = list(var_list)
  return var_list

def optimize(loss, grads_and_vars, summary_key=None, var_list=None,
             optimizer=None, use_prettytensor=False, global_step=None):
  """Initialize inference algorithm. It initializes hyperparameters
  and builds ops for the algorithm's computation graph.

  Args:
    optimizer: str or tf.train.Optimizer, optional.
      A TensorFlow optimizer, to use for optimizing the variational
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    use_prettytensor: bool, optional.
      `True` if aim to use PrettyTensor optimizer (when using
      PrettyTensor) or `False` if aim to use TensorFlow optimizer.
      Defaults to TensorFlow.
    global_step: tf.Variable, optional.
      A TensorFlow variable to hold the global step.
  """
  if summary_key is not None:
    tf.summary.scalar("loss", loss, collections=[summary_key])
    for grad, var in grads_and_vars:
      # replace colons which are an invalid character
      tf.summary.histogram("gradient/" +
                           var.name.replace(':', '/'),
                           grad, collections=[summary_key])
      tf.summary.scalar("gradient_norm/" +
                        var.name.replace(':', '/'),
                        tf.norm(grad), collections=[summary_key])

  if optimizer is None and global_step is None:
    # Default optimizer always uses a global step variable.
    global_step = tf.Variable(0, trainable=False, name="global_step")

  if isinstance(global_step, tf.Variable):
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               100, 0.9, staircase=True)
  else:
    learning_rate = 0.01

  # Build optimizer.
  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif isinstance(optimizer, str):
    if optimizer == 'gradientdescent':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer class not found:', optimizer)
  elif not isinstance(optimizer, tf.train.Optimizer):
    raise TypeError("Optimizer must be str, tf.train.Optimizer, or None.")

  with tf.variable_scope(None, default_name="optimizer") as scope:
    if not use_prettytensor:
      train_op = optimizer.apply_gradients(grads_and_vars,
                                           global_step=global_step)
    else:
      import prettytensor as pt
      # Note PrettyTensor optimizer does not accept manual updates;
      # it autodiffs the loss directly.
      train_op = pt.apply_optimizer(optimizer, losses=[loss],
                                      global_step=global_step,
                                      var_list=var_list)
  return train_op

def update(train_op, n_print, summarize=None, train_writer=None,
           debug=False, op_check=None, *args, **kwargs):
  """Run one iteration of optimization.

  Args:
    args: things like `loss`
    kwargs: things like 'feed_dict'
    feed_dict: dict, optional.
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

  Returns:
    dict.
    Dictionary of algorithm-specific information. In this case, the
    loss function value after one iteration.
  """
  # TODO
  # feed_dict = {}
  # for key, value in six.iteritems(data):
  #   if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
  #     feed_dict[key] = value
  sess = get_session()
  feed_dict = kwargs.pop('feed_dict', {})
  values = sess.run([train_op] + list(kwargs.values()), feed_dict)
  values = values[1:]

  if debug:
    sess.run(op_check, feed_dict)

  if summarize is not None and n_print != 0:
    if t == 1 or t % n_print == 0:
      summary = sess.run(summarize, feed_dict)
      train_writer.add_summary(summary, t)

  return dict(zip(kwargs.keys(), values))

def print_progress(progbar, n_print, info_dict):
  """Print progress to output."""
  if n_print != 0:
    t = info_dict['t']
    if t == 1 or t % n_print == 0:
      progbar.update(t, {'Loss': info_dict['loss']})
