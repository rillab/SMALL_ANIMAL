#CUDA_VISIBLE_DEVICES=0 python3 'file name'
# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
start_time = time.time()

from glandnet import *
import argparse
import sys
import math
import numpy as np
import os
import random
import tensorflow as tf
from skimage.io import imread
from skimage.io import imsave
#from skimage.transform import rotate

FLAGS = None

import matplotlib.pyplot as plt
import importlib as imp
import pandas as pd
import loaddata as ld
imp.reload(ld)

#!!!!! Remember to set nclass=2 in glandnet.py

#Directory settings
logdir = '../r20/info'
result_dir = '../out/r20_out/7000/'
test_dir = "../test_mask_r20"
model_loc = "C:/Users/pzhou10/Downloads/SPIE2020_stage2/640To256/r20/save/val3/7000"
nc = 32
batch_size=8 #4672Mib
#batch_size=16 #15516Mib
#batch_size=8 #8768Mib
#batch_size=2 #3136Mib
#batch_size=18 #15516Mib

imageType=3 # imageType for RGB image
ins=256 # input image size for NN, this should match the image size, because we are not doing cropping.
res = ins # in noCrop, we should set 'res' equals to 'ins'

# # Create the model
sess = tf.InteractiveSession()
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

input_img = tf.placeholder(tf.float32, [None,ins,ins,imageType])
phase = tf.placeholder(tf.bool, name='phase')

#tf.summary.image('input',input_img,batch_size)
with tf.variable_scope('scale1'):
    with tf.variable_scope('conv1'):
        c1_1_conv = conv_(input_img,3,2,imageType,nc,phase)
        c1_1_bn = bn_(c1_1_conv,phase)
        c1_1_relu = relu_(c1_1_bn)
    with tf.variable_scope('conv2'):
        c1_2_conv = conv_(c1_1_relu,3,1,nc,nc,phase)
        c1_2_bn = bn_(c1_2_conv,phase)
        c1_2_relu = relu_(c1_2_bn)
    with tf.variable_scope('conv3'):
        c1_3_conv = conv_(c1_2_relu,3,1,nc,nc,phase)
        c1_3_bn = bn_(c1_3_conv,phase)
        c1_3_relu = relu_(c1_3_bn)

with tf.variable_scope('scale2'):
    pool1 = max_pool_2x2(c1_3_relu)
    c2 = stack(pool1,nc,4*nc,3,phase)

with tf.variable_scope('scale3'):
    pool2 = downbottleneck(c2, 4*nc, 8*nc, phase)
    c3 = stack(pool2,8*nc,8*nc,3,phase)

with tf.variable_scope('scale4'):
    pool3 = downbottleneck(c3, 8*nc, 16*nc, phase)
    c4 = stack(pool3,16*nc,16*nc,5,phase)

with tf.variable_scope('up2'):
    up2 = deconvall(c2,4*nc,2,phase)

with tf.variable_scope('up3'):
    up3 = deconvall(c3,8*nc,3,phase)

with tf.variable_scope('up4'):
    up4 = deconvall(c4,16*nc,4,phase)

with tf.variable_scope('final'):
    f1 = tf.concat([up2 , up3 , up4],3)
    with tf.variable_scope('final_conv1'):
        f1_conv = conv_(f1,3,1,3*nclass,nclass,phase)
        f1_bn = bn_(f1_conv,phase)
        f1_relu = relu_(f1_bn)
    with tf.variable_scope('final_conv2'):
        output_nobias = conv_(f1_relu,1,1,nclass,nclass,phase)
        b=bias_variable([nclass])
        output = output_nobias + b

with tf.variable_scope('result'):
    # result = tf.nn.softmax(output) # for classification
    result = output # for regression

output_gt = tf.placeholder(tf.float32, [None, ins, ins, 2])
# Define loss and optimizer
squared_error = tf.squared_difference(output, output_gt)
sum_squared_error = tf.reduce_sum(squared_error, axis=[1,2,3])
l2_loss = tf.reduce_mean(sum_squared_error)

tf.summary.scalar('error',l2_loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer().minimize(l2_loss) # Original learning rate
    #train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)# learning rate after 80k iterations
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(logdir, sess.graph)

saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name],max_to_keep=None)

# # Select training modes
if (len(model_loc)==0):
    print('Init model from scratch')
    sess.run(tf.global_variables_initializer())
else:
    print('Load model: ' + model_loc)
    #saver = tf.train.import_meta_graph("C:/Users/pzhou10/Downloads/SPIE2020_stage2/640To256/r10/save/model37500/37500.ckpt.meta")
    #saver.restore(sess, model_loc)
    saver.restore(sess,tf.train.latest_checkpoint(model_loc))
    sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name])) #just initialize adam parameters

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print(total_parameters)

##training
def model_apply_threeClass_noCrop(sess,result,input_img,phase,val_image,val_label,ins):
    nrotate = 4
    val_out_rotate = np.zeros([1,ins,ins,nclass]) #nclass = 2
    error = 0
    val_label = val_label[:,:,0:2] #as the training require only 2 layers and the third layer is all 0
    for j in range(nrotate):
        feed_image=np.zeros([1,ins,ins,imageType]) #imageType = 3
        feed_label=np.zeros([1,ins,ins,nclass])
        feed_image[0,:,:,:] = np_rotate(val_image, j)
        feed_label[0,:,:,:] = np_rotate(val_label, j)
        val_out, val_out_error = sess.run([result,l2_loss],
                feed_dict={input_img:feed_image, output_gt:feed_label[:,:,:,:2], phase: False})

        val_out = val_out[0,:,:,:]
        val_out_rotate[0,:,:,:] = val_out_rotate[0,:,:,:] + np_rotate(val_out,4-j) #add 4 result and then average them
        error = error + val_out_error
    val_out_rotate = val_out_rotate / nrotate
    error = error / nrotate
    return val_out_rotate[0,:,:,:], error

test_loss = np.array([])

for mask_file in os.listdir(os.path.join(test_dir)):
    if mask_file.endswith("_mask.png"):
        img_file = mask_file[:-9] + ".png"
        val_13_label = imread(os.path.join(test_dir,mask_file))
        val_13_image = imread(os.path.join(test_dir,img_file))
        val_13_image = val_13_image/255
        val_13_label = val_13_label/255
        val_13_out, test_1_loss = model_apply_threeClass_noCrop(sess,result,input_img,phase,val_13_image,val_13_label,ins)
        test_loss = np.append(test_loss, test_1_loss)

        add = np.zeros(shape=(ins,ins))
        val_13_out_new = np.zeros(shape=(ins,ins,3))
        val_13_out_new[:,:,0:2] = val_13_out
        val_13_out_new[:,:,2] = add
        val_out_dir = result_dir + img_file
        imsave(val_out_dir, val_13_out_new)
        print(test_1_loss)

val_loss_dir = result_dir + 'test_loss'
np.save(val_loss_dir, test_loss)
print("--- %s seconds ---" %(time.time() - start_time))
