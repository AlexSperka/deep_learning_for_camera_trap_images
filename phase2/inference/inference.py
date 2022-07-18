#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import resnet
import cv2
import numpy as np
import json
from collections import Counter
import requests
from flask import Flask, request, redirect, jsonify, url_for, abort, make_response
import sys
import os
import base64

# constants
load_size = [256,256,3]
crop_size = [224,224,3]
batch_size = 512
num_classes = [48, 12, 2, 2, 2, 2, 2, 2]
num_channels = 3
samples = 1
num_batches = -1
batch_size = 1
if num_batches==-1:
    if(samples%batch_size==0):
      num_batches= int(samples/batch_size)
    else:
      num_batches= int(samples/batch_size)+1
num_threads = 20
depth = 152
ckpt_path = '/root/phase2'         # Path of the checkpoints after building the docker image

tf.compat.v1.disable_eager_execution()  #<--- Disable eager execution

def _test_preprocess(reshaped_image, crop_size, num_channels):

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_with_crop_or_pad(reshaped_image, crop_size[0], crop_size[1])
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)
  # Set the shapes of tensors.
  float_image.set_shape([crop_size[0], crop_size[1], num_channels])

  return float_image

def decode_img(img_str):
  img_bytes = bytes(img_str, 'utf-8')
  img_buff = base64.b64decode(img_bytes)
  img_jpg = np.frombuffer(img_buff, dtype=np.uint8)
  img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
  return img  

def read_img(image,desired_width = 256, desired_height = 256):
  img = cv2.resize(image, (desired_width, desired_height))
  _, img = cv2.imencode('.jpg', img)
  img = img.tostring()
  return img     

def inference_model(image,input_img):
  img = read_img(input_img)
  top5guesses_id, top5conf, top3guesses_cn, top3conf, top1guesses_bh, top1conf = sess.run([top5ind_id, top5val_id, top3ind_cn, top3val_cn, top1ind_bh, top1val_bh], feed_dict = {image: img})
  
  activity = np.zeros(6)
  confidence_activity = np.zeros(6)
  
  count = top3guesses_cn
  confidence_count = top3conf
  
  id = top5guesses_id
  confidence_id = top5conf
  
  for i in range(0,6):
    activity[i] = top1guesses_bh[i][0][0]
    confidence_activity[i] = top1conf[i][0][0]
  return id, confidence_id, count, confidence_count, activity, confidence_activity 

app = Flask(__name__)
app.config['DEBUG'] = True

g = tf.Graph().as_default()
tf.device('/cpu:0')
# Get images and labels.
image = tf.compat.v1.placeholder(tf.string, name='input')
reshaped_image = tf.cast(tf.image.decode_jpeg(image, channels = num_channels), tf.float32)
reshaped_image = tf.image.resize(reshaped_image, (load_size[0], load_size[1]))
reshaped_image = _test_preprocess(reshaped_image, crop_size, num_channels)
imgs = reshaped_image[None, ...]
# Performing computations on a GPU
tf.device('/gpu:0')
# Build a Graph that computes the logits predictions from the
# inference model.
logits = resnet.inference(imgs, depth, num_classes, 0.0, False)
top5_id = tf.nn.top_k(tf.nn.softmax(logits[0]), 5)
top5ind_id= top5_id.indices
top5val_id= top5_id.values
# Count
top3_cn = tf.nn.top_k(tf.nn.softmax(logits[1]), 3)
top3ind_cn= top3_cn.indices
top3val_cn= top3_cn.values
# Additional Attributes (e.g. description) == behavior
top1_bh= [None]*6
top1ind_bh= [None]*6
top1val_bh= [None]*6

for i in range(0,6):
  top1_bh[i]= tf.nn.top_k(tf.nn.softmax(logits[i+2]), 1)
  top1ind_bh[i]= top1_bh[i].indices
  top1val_bh[i]= top1_bh[i].values
  
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())  
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())
ckpt = tf.train.get_checkpoint_state(ckpt_path)
print(ckpt_path)
print(ckpt)
if ckpt: #and ckpt.model_checkpoint_path:
# Restores from checkpoint
  saver.restore(sess, ckpt.model_checkpoint_path)
  print('pass')
else:
  print('error')  

def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')

@app.route('/model/api/v1.0/recognize', methods=['POST'])
def recognize_activity():
  img=None
  if not request.json or not 'img_path' in request.json:
    abort(204)
  img = cv2.imread(request.json['img_path'])    # Add the image path, ex: '/home/animal.jpg'
  if(img is not None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_encoded=str(encode_img(img))
    img_decoded=decode_img(img_encoded)

    id, confidence_id, count, confidence_count, activity, confidence_activity  = inference_model(image,img_decoded)
    return make_response(jsonify({'Status: ': 'finished', 'id': json.dumps(id.tolist()), 'confidence_id': json.dumps(confidence_id.tolist()), 'count': json.dumps(count.tolist()), 'confidence_count': json.dumps(confidence_count.tolist()), 'activity': json.dumps(activity.tolist()), 'confidence_activity': json.dumps(confidence_activity.tolist())}), 200)   

  
if __name__ == '__main__':
  app.run(host='0.0.0.0')    
  
