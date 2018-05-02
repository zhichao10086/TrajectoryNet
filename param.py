from enum import Enum
import tensorflow as tf

WIDTH = 20

train_file_pattern = "interval_[1-5]_train_*.tfrecords"
valid_file_pattern = "interval_[1-5]_valid_*.tfrecords"
test_file_pattern = "interval_[1-5]_test_*.tfrecords"


SPEED_SEC = "speed_sec"
ACC_SEC = "acc_sec"
AVG_SPEED = "avg_speed"
STD_SPEED = "std_speed"
MEAN_ACC = "mean_acc"
STD_ACC = "std_acc"
HEAD = "head"
HEAD_MEAN = "head_mean"
STD_HEAD = "std_head"
EARLY = "early"
LABEL = "label"

feature = {
        SPEED_SEC: tf.FixedLenFeature([],tf.string),
        AVG_SPEED : tf.FixedLenFeature([],tf.string),
        STD_SPEED : tf.FixedLenFeature([],tf.string),
        ACC_SEC   : tf.FixedLenFeature([],tf.string),
        MEAN_ACC  : tf.FixedLenFeature([],tf.string),
        STD_ACC   : tf.FixedLenFeature([],tf.string),
        HEAD      : tf.FixedLenFeature([],tf.string),
        HEAD_MEAN : tf.FixedLenFeature([],tf.string),
        STD_HEAD  : tf.FixedLenFeature([],tf.string),
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