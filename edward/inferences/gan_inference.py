from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (process_inputs, transform,
    process_dict, process_var_list)
from edward.util import get_session


def gan_inference(latent_vars=None, data=None, discriminator=None,
                  auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  #### Notes

  `GANInference` does not support latent variable inference. Note
  that GAN-style training also samples from the prior: this does not
  work well for latent variables that are shared across many data
  points (global variables).

  In building the computation graph for inference, the
  discriminator's parameters can be accessed with the variable scope
  "Disc".

  GANs also only work for one observed random variable in `data`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  #### Examples

  ```python
  z = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
  x = generative_network(z)

  inference = ed.GANInference({x: x_data}, discriminator)
  ```
  """
  """Create an inference algorithm.

  Args:
    data: dict.
      Data dictionary which binds observed variables (of type
      `RandomVariable` or `tf.Tensor`) to their realizations (of
      type `tf.Tensor`).  It can also bind placeholders (of type
      `tf.Tensor`) used in the model to their realizations.
    discriminator: function.
      Function (with parameters) to discriminate samples. It should
      output logit probabilities (real-valued) and not probabilities
      in $[0, 1]$.
    var_list: list of tf.Variable, optional.
      List of TensorFlow variables to optimize over (in the generative
      model). Default is all trainable variables that `latent_vars`
      and `data` depend on.
  """
  if not callable(discriminator):
    raise TypeError("discriminator must be a callable function.")
  latent_vars, data = process_inputs(None, data)
  # latent_vars, _ = transform(latent_vars, auto_transform)
  scale = process_dict(scale)
  var_list = process_var_list(var_list, latent_vars, data)

  x_true = list(six.itervalues(data))[0]
  x_fake = list(six.iterkeys(data))[0]
  with tf.variable_scope("Disc"):
    d_true = discriminator(x_true)

  with tf.variable_scope("Disc", reuse=True):
    d_fake = discriminator(x_fake)

  if summary_key is not None:
    tf.summary.histogram("discriminator_outputs",
                         tf.concat([d_true, d_fake], axis=0),
                         collections=[summary_key])

  reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
  reg_terms_all = tf.losses.get_regularization_losses()
  reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

  loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_true), logits=d_true) + \
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(d_fake), logits=d_fake)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_fake), logits=d_fake)
  loss_d = tf.reduce_mean(loss_d) + tf.reduce_sum(reg_terms_d)
  loss = tf.reduce_mean(loss) + tf.reduce_sum(reg_terms)

  var_list_d = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
  if var_list is None:
    var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

  grads_d = tf.gradients(loss_d, var_list_d)
  grads = tf.gradients(loss, var_list)
  grads_and_vars_d = list(zip(grads_d, var_list_d))
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars, loss_d, grads_and_vars_d
  # TODO when users call this, this has duplicate for summary.scalar("loss")
  # train = optimize(loss, grads_and_vars, summary_key)
  # train_d = optimize(loss_d, grads_and_vars_d, summary_key)

# TODO how to pass in update and print_progress to run()
# + have this be default for gan_inference, wgan_inference,
# implicit_klqp, bigan_inference
# + use registration mechanisms on objects? (but still, this is just
# train ops)
#   + for now, use string update modes for run(); put all these inside
#   inference.py?
def update(train_op, train_op_d, n_print, summarize=None, train_writer=None,
           debug=False, op_check=None, variables=None, *args, **kwargs):
  """Run one iteration of optimization.

  Args:
    variables: str, optional.
      Which set of variables to update. Either "Disc" or "Gen".
      Default is both.

  Returns:
    dict.
    Dictionary of algorithm-specific information. In this case, the
    iteration number and generative and discriminative losses.

  #### Notes

  The outputted iteration number is the total number of calls to
  `update`. Each update may include updating only a subset of
  parameters.
  """
  # if feed_dict is None:
  #   feed_dict = {}
  # for key, value in six.iteritems(self.data):
  #   if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
  #     feed_dict[key] = value
  sess = get_session()
  feed_dict = kwargs.pop('feed_dict', {})
  if variables is None:
    values = sess.run([train_op, train_op_d] + list(kwargs.values()), feed_dict)
    values = values[2:]
  elif variables == "Gen":
    kwargs['loss_d'] = 0.0
    values = sess.run([train_op] + list(kwargs_temp.values()), feed_dict)
    values = values[1:]
  elif variables == "Disc":
    kwargs['loss'] = 0.0
    values = sess.run([train_op_d] + list(kwargs_temp.values()), feed_dict)
    values = values[1:]
  else:
    raise NotImplementedError("variables must be None, 'Gen', or 'Disc'.")

  if debug:
    sess.run(op_check, feed_dict)

  if summarize is not None and n_print != 0:
    if t == 1 or t % self.n_print == 0:
      summary = sess.run(summarize, feed_dict)
      train_writer.add_summary(summary, t)

  return dict(zip(kwargs_temp.keys(), values))

def print_progress(progbar, n_print, info_dict):
  """Print progress to output.
  """
  if n_print != 0:
    t = info_dict['t']
    if t == 1 or t % n_print == 0:
      progbar.update(t, {'Gen Loss': info_dict['loss'],
                         'Disc Loss': info_dict['loss_d']})
