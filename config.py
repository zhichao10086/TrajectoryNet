import json
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from enum import Enum
from param import RNNType
from param import NetType


class Config(object):
    def __init__(self,configFile="data/config.json"):
        dconf = json.load(open(configFile))
        #测试集ID
        self.test_id = dconf['test_id']
        #验证集ID
        self.val_id = dconf['val_id']
        #预期序列长度
        self.exp_seq_len = dconf["exp_seq_len"]
        #学习速率
        self.learning_rate = dconf["learning_rate"]
        #批数据尺寸
        self.batch_size = dconf["batch_size"]
        #隐藏层数
        self.num_layers = dconf["num_layers"]
        #迭代周期
        self.num_epochs = dconf["num_epochs"]
        #是否开启tensorboard
        self.tensorboard = dconf["tensorboard"]
        self.init_scale = dconf["init_scale"]
        #线程数
        self.num_threads = dconf["num_threads"]
        #gru中的隐藏节点
        self.hidden_size = dconf["hidden_size"]
        #任务名称
        self.task = dconf["task"]
        #是否用GPU加速
        self.useGPU = dconf["useGPU"]
        #weight初始化方式
        self.weight_initializer = dconf["weight_initializer"]
        #评估频率
        self.evaluate_freq = dconf["evaluate_freq"]
        self.testmode = dconf["testmode"]
        #是否有检查点
        self.checkpoint = dconf["checkpoint"]
        #是否重载变量
        self.restore = dconf["restore"]
        #激励函数
        self.activation = dconf["activation"]
        self.test_mode = dconf["test_mode"]
        #分类个数
        self.num_classes = dconf["num_classes"]
        #maxout中的单元个数
        self.maxOut_numUnits = dconf["maxOut_numUnits"]
        #特征数量
        self.num_features = dconf["num_features"]
        #嵌入后的向量维度
        self.embeded_dims = dconf["embeded_dims"]
        #L2正则化超参数
        self.l2_preparam = dconf["l2_preparam"]
        #
        self.rnn_type = dconf["rnn_type"]





class TrainingConfig(object):
    def __init__(self,is_training,is_validation,is_test,batch_size,len_features,net_type = NetType.RNN_NV1,rnn_type = RNNType.GRU):
        self.is_training = is_training
        self.is_validation = is_validation
        self.is_test = is_test
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.net_type = net_type

        #特征长度即onehot总长度
        self.len_features = len_features
        self.train_seq_len = []
        self.val_seq_len = []
        self.test_seq_len = []
        self.activation = tanh

