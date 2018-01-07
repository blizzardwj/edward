#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical


def klqp(model, variational, align_latent, align_data, *args):
  """Loss function equal to KL(q||p) up to a constant.

  Args:
    model: function whose inputs are a subset of `args` (e.g., for
      discriminative). Output is not used.
    variational: function whose inputs are a subset of `args` (e.g.,
      for amortized). Output is not used.
    align_latent: function of string, aligning `model` latent
      variables with `variational`. It takes a model variable's name
      as input and returns a string, indexing `variational`'s trace;
      else identity.
    align_data: function of string, aligning `model` observed
      variables with data. It takes a model variable's name as input
      and returns an integer, indexing `args`; else identity.
    args: data inputs. It is passed at compile-time in Graph
      mode or runtime in Eager mode.
  """
  def _intercept(f, *args, **kwargs):
    """Set model's sample values to variational distribution's and data."""
    name = kwargs.get('name', None)
    if isinstance(align_data(name), int):
      kwargs['value'] = arg[align_data(name)]
    else:
      kwargs['value'] = posterior_trace[align_latent(name)].value
    return f(*args, **kwargs)
  with Trace() as posterior_trace:
    call_function_up_to_args(variational, args)
  with Trace(intercept=_intercept) as model_trace:
    call_function_up_to_args(model, args)

  log_p = tf.reduce_sum([tf.reduce_sum(x.log_prob(x.value))
                         for x in model_trace.values()
                         if isinstance(x, tfd.Distribution)])
  log_q = tf.reduce_sum([tf.reduce_sum(qz.log_prob(qz.value))
                         for qz in posterior_trace.values()
                         if isinstance(qz, tfd.Distribution)])
  loss = log_q - log_p
  return loss


def call_function_up_to_args(f, args):
  import inspect
  if hasattr(f, "_func"):  # make_template()
    f = f._func
  num_args = len(inspect.getargspec(f).args)
  if num_args > 0:
    return f(args[:num_args])
  return f()


ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
p = Beta(1.0, 1.0)
x = Bernoulli(probs=p, sample_shape=10)

# INFERENCE
def posterior(x=None):
  qp = Empirical(tf.Variable(tf.zeros([1000]) + 0.5), name="p")
  return qp

def proposal(x=None):
  proposal_p = Beta(3.0, 9.0, name="proposal_p")
  return proposal_p

# TODO update? maybe just transition of the posterior chain
# `update` is a function of the realized data (tfe.Tensor) and returns
# a train operation.
# ed.automate(train_op, x_data)
update = ed.metropolis_hastings(
    model,
    posterior,
    proposal,
    latent_align=lambda name: "qp" if name == "p" else name,
    data_align=lambda name: 0 if name == "x" else name,
    proposal_align=lambda name: "proposal_p" if name =="qp" else name,
    x_data)
# update = ed.hmc(
#   model,
#   posterior,
#   data)

sess = tf.Session()  # ed.get_session() not needed
tf.global_variables_initializer().run()  # presumably not needed
for _ in range(1000):
  info_dict = update(x_data)
