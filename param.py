from enum import Enum
import tensorflow as tf

WIDTH = 30

train_file_pattern = "interval_[1-5]_label_[0-3]_train.tfrecords"
valid_file_pattern = "interval_[1-5]_label_[0-3]_valid.tfrecords"
test_file_pattern = "interval_[1-5]_label_[0-3]_test.tfrecords"


SPEED_SEC = "speed_sec"
ACC_SEC = "acc_sec"
AVG_SPEED = "avg_speed"
STD_SPEED = "std_speed"
MEAN_ACC = "mean_acc"
STD_ACC = "std_acc"
EARLY = "early"
LABEL = "label"

feature = {
        SPEED_SEC: tf.FixedLenFeature([],tf.string),
        AVG_SPEED : tf.FixedLenFeature([],tf.string),
        STD_SPEED : tf.FixedLenFeature([],tf.string),
        ACC_SEC   : tf.FixedLenFeature([],tf.string),
        MEAN_ACC  : tf.FixedLenFeature([],tf.string),
        STD_ACC   : tf.FixedLenFeature([],tf.string),
        EARLY   :   tf.FixedLenFeature([],tf.int64),
        LABEL:tf.FixedLenFeature([],tf.int64)
    }

class RNNType(Enum):
    LSTM = 1 # LSTM unidirectional
    LSTM_b = 2 # LSTM bidirectional
    GRU = 3 # GRU
    GRU_b = 4 # GRU, bidirectional

class NetType(Enum):
    DNN  = 1
    CNN  = 2
    RNN_NV1 = 3
    RNN_NVN = 4

class FeatureName(Enum):
    SPEED_SEC = "speed_sec"
    ACC_SEC = "acc_sec"
    AVG_SPEED = "avg_speed"
    STD_SPEED = "std_speed"
    MEAN_ACC = "mean_acc"
    STD_ACC = "std_acc"