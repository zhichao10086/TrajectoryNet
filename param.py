from enum import Enum

width = 30
SPEED_SEC = "speed_sec"
ACC_SEC = "acc_sec"
AVG_SPEED = "avg_speed"
STD_SPEED = "std_speed"
MEAN_ACC = "mean_acc"
STD_ACC = "std_acc"
LABEL = "label"



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