from enum import Enum

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