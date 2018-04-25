import json
import numpy as np
import random
import math
import config
import pandas as pd
from param import FeatureName
from param import RNNType
import tensorflow as tf
import os
import param

a = np.random.randint(0,10,[2,2])
b = np.random.randint(0,10,[2,2])
c = np.random.randint(0,10,[2,2])

a_df = pd.DataFrame(a)
b_df = pd.DataFrame(b)
c_df = pd.DataFrame(c)
cv = pd.concat([a_df,b_df,c_df],axis=1)
print(24%10)

# data_dir ="G:/all_data/tfrecords/"
# filenames = os.listdir(data_dir)
# filenames = [os.path.join(data_dir,i) for i in filenames]
#
# feature = {
#     FeatureName.SPEED_SEC.value : tf.FixedLenFeature([],tf.string),
#     FeatureName.AVG_SPEED.value : tf.FixedLenFeature([],tf.string),
#     FeatureName.STD_SPEED.value : tf.FixedLenFeature([],tf.string),
#     FeatureName.ACC_SEC.value   : tf.FixedLenFeature([],tf.string),
#     FeatureName.MEAN_ACC.value  : tf.FixedLenFeature([],tf.string),
#     FeatureName.STD_ACC.value   : tf.FixedLenFeature([],tf.string),
#     "label":tf.FixedLenFeature([],tf.int64)
# }
#
# filename_queue = tf.train.string_input_producer(filenames,num_epochs=1)
# reader = tf.TFRecordReader()
# _,serialized_example = reader.read(filename_queue)
#
# features = tf.parse_single_example(serialized_example,features= feature)
# speed_sec_flat = tf.decode_raw(features[param.SPEED_SEC],tf.int64)
# speed_sec = tf.reshape(speed_sec_flat,[-1,param.width])
# label = tf.cast(features[param.LABEL],tf.int64)
#
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(2):
#         speed_sec1,label1 = sess.run([speed_sec,label])
#         print(type(speed_sec1))
#         full_seq_num = speed_sec1.shape[0] // 100
#         print(full_seq_num)
#         list = []
#         begin = 0
#         if full_seq_num >0:
#             for e in range(full_seq_num):
#                 begin = e*30
#                 list.append(tf.slice(speed_sec1,[begin,30],[100,30]))
#         remain = speed_sec1.shape[0] - full_seq_num*100
#         remain_tensor = tf.slice(speed_sec1,[begin,30],[remain,30])
#
#         remain_tensor_pad = tf.pad(remain_tensor,[[0,100 - remain],[0,0]])
#         list.append(remain_tensor_pad)
#         for h in list:
#             print(h)
#
#         tf.train.shuffle_batch()
#
#
#
#     coord.request_stop()
#     coord.join(threads)
