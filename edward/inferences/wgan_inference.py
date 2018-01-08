from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (process_inputs, transform,
    process_dict, process_var_list)
from edward.inferences import gan_inference
from edward.util import get_session

try:
  from edward.models import Uniform
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def wgan_inference(latent_vars=None, data=None, discriminator=None,
                   penalty=10.0, clip=None,
                   auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative], using the Wasserstein distance
  [@arjovsky2017wasserstein].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  #### Notes

  Argument-wise, the only difference from `GANInference` is
  conceptual: the `discriminator` is better described as a test
  function or critic. `WGANInference` continues to use
  `discriminator` only to share methods and attributes with
  `GANInference`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  #### Examples

  ```python
  z = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
  x = generative_network(z)

  inference = ed.WGANInference({x: x_data}, discriminator)
  ```
  """
  """Initialize inference algorithm. It initializes hyperparameters
  and builds ops for the algorithm's computation graph.

  Args:
    penalty: float, optional.
      Scalar value to enforce gradient penalty that ensures the
      gradients have norm equal to 1 [@gulrajani2017improved]. Set to
      None (or 0.0) if using no penalty.
    clip: float, optional.
      Value to clip weights by. Default is no clipping.
  """
  clip_op = None
  if clip is not None:
    var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    clip_op = [w.assign(tf.clip_by_value(w, -clip, clip)) for w in var_list]

  latent_vars, data = process_inputs(None, data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  scale = process_dict(scale)
  var_list = process_var_list(var_list, latent_vars, data)

  x_true = list(six.itervalues(data))[0]
  x_fake = list(six.iterkeys(data))[0]
  with tf.variable_scope("Disc"):
    d_true = discriminator(x_true)

  with tf.variable_scope("Disc", reuse=True):
    d_fake = discriminator(x_fake)

  if penalty is None or penalty == 0:
    penalty = 0.0
  else:
    eps = Uniform().sample(x_true.shape[0])
    while eps.shape.ndims < x_true.shape.ndims:
      eps = tf.expand_dims(eps, -1)
    x_interpolated = eps * x_true + (1.0 - eps) * x_fake
    with tf.variable_scope("Disc", reuse=True):
      d_interpolated = discriminator(x_interpolated)

    gradients = tf.gradients(d_interpolated, [x_interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                   list(range(1, gradients.shape.ndims))))
    penalty = penalty * tf.reduce_mean(tf.square(slopes - 1.0))

  reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
  reg_terms_all = tf.losses.get_regularization_losses()
  reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

  mean_true = tf.reduce_mean(d_true)
  mean_fake = tf.reduce_mean(d_fake)
  loss_d = mean_fake - mean_true + penalty + tf.reduce_sum(reg_terms_d)
  loss = -mean_fake + tf.reduce_sum(reg_terms)

  var_list_d = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
  if var_list is None:
    var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

  grads_d = tf.gradients(loss_d, var_list_d)
  grads = tf.gradients(loss, var_list)
  grads_and_vars_d = list(zip(grads_d, var_list_d))
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars, loss_d, grads_and_vars_d

# TODO set to use this for wgan_inference
def update(clip_op, variables=None, *args, **kwargs):
  info_dict = gan_inference.update(variables=variables, *args, **kwargs)

  sess = get_session()
  if clip_op is not None and variables in (None, "Disc"):
    sess.run(clip_op)

  return info_dict
