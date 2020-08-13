from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import numpy as np
import os
import random

import tensorflow as tf

import skimage
from skimage.io import imread
from skimage.io import imsave
#from skimage.transform import rotate

FLAGS = None

imageType = 3
ous = 256
ins = 256
#interv = 200
nclass = 2
#nclass = 5 #<--! multiclass training
batch_size = 4
#homedir = '/home/ubuntu/lab/projects/SPIE2018/'
#logdir = homedir + 'tftrain'
#resultdir = homedir + 'tfsave_gland'
#imgdir = homedir + 'Documents/Mouse2017/workplace/'

def load_data(file_dir):
  train_imgs = []
  label_imgs = []
  for file in os.listdir(os.path.join(file_dir,'img/')):
    if file.endswith(".png"):
      tI=imread(os.path.join(file_dir,'img/',file))
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      train_imgs.append(tI)
      lI=imread(os.path.join(file_dir,'mask/',file))
      label_imgs.append(lI)
  return train_imgs, label_imgs

def load_data_multitask(file_dir):
  train_imgs = []
  label_imgs = []
  for file in os.listdir(os.path.join(file_dir,'img/')):
    if file.endswith(".png"):
      tI=imread(os.path.join(file_dir,'img/',file))
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      train_imgs.append(tI)
      lI=np.load(os.path.join(file_dir,'mask/',file[:-4]+'.data.npy'))
      label_imgs.append(lI)
  return train_imgs, label_imgs

def np_rotate(data, times):
    result = data.copy()
    for i in range(times):
        result = np.rot90(result)
    return result

def get_image(train_imgs,label_imgs):
  i_idx = random.randint(0, len(train_imgs)-1)
  big_tI = train_imgs[i_idx]
  big_lI = label_imgs[i_idx]
  xs = big_tI.shape[0]
  ys = big_tI.shape[1]
  stx = random.randint(0,xs-ins)
  sty = random.randint(0,ys-ins)
  train_sample = np.zeros((ins,ins,imageType))
  label_sample = np.zeros((ins,ins))
  train_sample = big_tI[stx:stx+ins,sty:sty+ins,:]/255
  label_sample = big_lI[stx:stx+ins,sty:sty+ins]
  nrotate = random.randint(0, 3)
  train_sample = np_rotate(train_sample, nrotate)
  label_sample = np_rotate(label_sample, nrotate)
  #label_sample = np.round(np_rotate(label_sample, 90*nrotate)*255).astype('uint8')
  #print("--> get_image: (xs, ys) is ({},{}), (stx, sty) is ({},{}), label shape is {}".format(xs,ys,stx,sty, big_lI.shape))#<--------Zheng
  return train_sample, label_sample

def get_batch(train_imgs,label_imgs,batch_size):
  train_samples = np.zeros((batch_size,ins,ins,imageType))
  label_samples = np.zeros((batch_size,ins,ins))
  for i in range(batch_size):
    train_sample, label_sample = get_image(train_imgs,label_imgs)
    #print('--> get_batch:', i,label_sample.shape)#<--------Zheng
    train_samples[i,:,:,:] = train_sample
    #label_samples[i,:,:] = label_sample/interv
    label_samples[i,:,:] = label_sample #<-----Zheng's modify for 3 classes' classification
  return train_samples, label_samples

def weight_variable(shape,stdv):
  initial = tf.get_variable("weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=stdv))
  return initial

def bias_variable(shape):
  initial = tf.get_variable("bias", shape=shape, initializer=tf.constant_initializer(value=0.1))
  return initial

def conv3(x, k, strides, nin, nout, phase):
  shape = [k,k,nin,nout]
  stdv = math.sqrt(2/(nin*3*3))
  w = weight_variable(shape, stdv)
  conv_result = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
  #if bias:
  #  b = bias_variable([nout])
  #  conv_result = conv_result + b
  bn_result = tf.contrib.layers.batch_norm(conv_result,
                                           center=True, scale=True,
                                           is_training=phase,
                                           scope='bn')
  #with tf.variable_scope('bn', reuse=True):
  #  moving_av = tf.get_variable('moving_mean')
  #  tf.summary.scalar(moving_av.name,tf.reduce_mean(moving_av))

  relu_result = tf.nn.relu(bn_result)
  return relu_result

def conv_(x,k,s,nin,nout,phase):
  #k: filter height and width
  #s: strides of the filter in height and width direction
  #x: tensor to be convoluted
  #nin: in_channels; nout: out_channels
  shape = [k,k,nin,nout]
  stdv = math.sqrt(2/(nin*3*3))
  w = weight_variable(shape, stdv)
  strides = [1,s,s,1]
  return tf.nn.conv2d(x, w, strides=strides, padding='SAME')

def bn_(x,phase):
  bn_result = tf.contrib.layers.batch_norm(x,
                                           center=True, scale=True,
                                           is_training=phase,
                                           decay = 0.9,
                                           scope='bn')
  return bn_result

def relu_(x):
  return tf.nn.relu(x)

def bottleneck(x, nin, nout, phase):
  if nin != nout:
    skip = tf.pad(x, [[0,0],[0,0],[0,0],[0,nout-nin]], 'CONSTANT')
  else:
    skip = x
  with tf.variable_scope('conv1'):
    c1_bn = bn_(x,phase)
    c1_relu = relu_(c1_bn)
    c1_conv = conv_(c1_relu,1,1,nin,nout/4,phase)

  with tf.variable_scope('conv2'):
    c2_bn = bn_(c1_conv,phase)
    c2_relu = relu_(c2_bn)
    c2_conv = conv_(c2_relu,3,1,nout/4,nout/4,phase)

  with tf.variable_scope('conv3'):
    c3_bn = bn_(c2_conv,phase)
    c3_relu = relu_(c3_bn)
    c3_conv = conv_(c3_relu,1,1,nout/4,nout,phase)

  out = skip + c3_conv
  return out

def downbottleneck(x, nin, nout, phase):
  #bottleneck layer that also has stride = 2 (downsample image resolution)
  input_shape = tf.shape(x)
  dx = tf.strided_slice(x,[0,0,0,0],input_shape,[1,2,2,1])
  if nin != nout:
    skip = tf.pad(dx, [[0,0],[0,0],[0,0],[0,nout-nin]], 'CONSTANT')
  else:
    skip = dx

  with tf.variable_scope('conv1'):
    c1_bn = bn_(x,phase)
    c1_relu = relu_(c1_bn)
    c1_conv = conv_(c1_relu,1,2,nin,nout/4,phase)

  with tf.variable_scope('conv2'):
    c2_bn = bn_(c1_conv,phase)
    c2_relu = relu_(c2_bn)
    c2_conv = conv_(c2_relu,3,1,nout/4,nout/4,phase)

  with tf.variable_scope('conv3'):
    c3_bn = bn_(c2_conv,phase)
    c3_relu = relu_(c3_bn)
    c3_conv = conv_(c3_relu,1,1,nout/4,nout,phase)

  out = skip + c3_conv
  return out

def stack(x, nin, nout, nblock, phase):
  for i in range(nblock):
    with tf.variable_scope('block%d' % (i)):
      if i==0:
        x = bottleneck(x,nin,nout,phase)
      else:
        x = bottleneck(x,nout,nout,phase)
  return x

def show(x,k):
  x_sm = tf.nn.softmax(x)
  x_sl = tf.unstack(x_sm,num=nclass,axis=-1)[k]
  return tf.expand_dims(x_sl,-1)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def deconv_(x, nin, nout, phase):
  shape = [4,4,nout,nin]
  stdv = math.sqrt(2/(nin*2*2))
  w = weight_variable(shape, stdv)
  input_shape = tf.shape(x)
  newshape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], nout])
  conv_result = tf.nn.conv2d_transpose(x, w, output_shape=newshape, strides=[1, 2, 2, 1], padding='SAME')
  bn_prepare = tf.reshape(conv_result, shape = newshape)
  bn_result = bn_(bn_prepare,phase)
  relu_result = relu_(bn_result)
  return relu_result

def deconv1_(x, nin, num_up, phase):
  cur_in = nin
  cur_out=(2**(num_up-1))

  for i in range(num_up):
    with tf.variable_scope('up%d' % (i)):
      x = deconv_(x,cur_in,cur_out,phase)
    cur_in = cur_out
    cur_out = math.floor(cur_out/2)

  return x

def deconvall(x, nin, num_up, phase):
  all_up = []
  for i in range(nclass):
    with tf.variable_scope('class%d' % (i)):
      result = deconv1_(x,nin,num_up,phase)
    all_up.append(result)
  cat = tf.concat(all_up,3)
  return cat
