import tensorflow as tf
import numpy as np
from math import ceil
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def batch_norm_wrapper(x, n_out, phase_train):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    #print(batch_mean, batch_var)
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm_wrapper_ff(x, n_out, phase_train):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    #print(batch_mean, batch_var)
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def deconv_layer_2lines(inputT, shape, train_phase, activation=True, name=None):
    wc = tf.Variable(tf.random_normal([shape[0], shape[1], shape[2], shape[3]], mean=0.0, stddev=0.01), name=name)
    deconv = tf.nn.conv2d_transpose(inputT, wc, strides=[1, 1, 1, 1], padding='SAME')    
    return deconv

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    wc = tf.Variable(tf.random_normal([shape[0], shape[1], shape[2], shape[3]], mean=0.0, stddev=0.01), name=name)
    conv = tf.nn.conv2d(inputT, wc, strides=[1, 1, 1, 1], padding='SAME')    
    bn = batch_norm_wrapper(conv, shape[3], train_phase)
    conv_out = tf.nn.relu(bn)
    return conv_out

def full_layer_with_bn(inputT, shape, train_phase, name=None):    
    wf = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), name=name+'_w')
    bf = tf.Variable(tf.zeros([shape[1]]), name=name+'_b')
    dense = tf.reshape(inputT, [-1, shape[0]])
    full = tf.add(tf.matmul(dense, wf),bf)
    bnf = batch_norm_wrapper_ff(full, shape[1], train_phase)
    #full_out = tf.nn.relu(bnf)
    full_out = tf.nn.tanh(bnf)
    return full_out


def full_layer(inputT, shape, name=None):    
    wf = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), name=name+'_w')
    bf = tf.Variable(tf.zeros([shape[1]]), name=name+'_b')
    full_out = tf.add(tf.matmul(inputT, wf), bf)    
    return full_out
    
    
def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)


def deconv_layer_koo(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def dump_unravel(indices, shape):
    """
    self-implented unravel indice, missing gradients, need fix
    """
    N = indices.get_shape().as_list()[0]
    tb = tf.constant([shape[0]], shape=[1,N])
    ty = tf.constant([shape[1]], shape=[1,N])
    tx = tf.constant([shape[2]], shape=[1,N])
    tc = tf.constant([shape[3]], shape=[1,N])

    c = indices % tc
    x = ((indices - c) // tc ) % tx
    t_temp = ((indices - c) // tc)
    y = ((t_temp - x) // tx) % ty
    t_temp = ((t_temp - x) // tx)
    b = (t_temp - y) // ty

    t_new = tf.transpose(tf.reshape(tf.pack([b,y,x,c]), (4, N)))
    return t_new

def upsample_with_pool_indices(value, indices, shape=None, scale=2, out_w=None, out_h=None,name="up"):
    s = shape.as_list()
    b = s[0]
    w = s[1]
    h = s[2]
    c = s[3]
    if out_w is not None:
      unraveled = dump_unravel(tf.to_int32(tf.reshape(indices,[b*w*h*c])), [b, out_w, out_h, c])
      ts = tf.SparseTensor(indices=tf.to_int64(unraveled), values=tf.reshape(value, [b*w*h*c]), shape=[b,out_w,out_h,c])
    else:
      unraveled = dump_unravel(tf.to_int32(tf.reshape(indices,[b*w*h*c])), [b, w*scale, h*scale, c])
      ts = tf.SparseTensor(indices=tf.to_int64(unraveled), values=tf.reshape(value, [b*w*h*c]), shape=[b,w*scale,h*scale,c])

    t_dense = tf.sparse_tensor_to_dense(ts, name=name, validate_indices=False)
    return t_dense
