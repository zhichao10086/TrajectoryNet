import config
import tensorflow as tf
from customized_gru import CustomizedGRU as GRUCell
from tensorflow.python.ops.rnn_cell import GRUCell as BasicGRUCell
import tensorflow.contrib as tf_ct
from tensorflow.contrib.rnn import BasicLSTMCell
from param import RNNType
from param import NetType
import linear
import param
import util




class Model(object):

    def __init__(self,conf,config):

        self.init_conf(conf)
        self.init_config(config)

        self.current_step = tf.Variable(0,trainable=False)
        self._learning_rate  = tf.train.exponential_decay(self.learning_rate,self.current_step,decay_steps=600,decay_rate=0.98,staircase=True)


        #self.current_step = tf.Variable(0)
        if self.net_type == NetType.DNN:
            self.init_dnn_type()
        elif self.net_type == NetType.CNN:
            self.init_cnn_type()
        elif self.net_type == NetType.RNN_NV1:
            self.init_rnn_type_nv1()
        elif self.net_type == NetType.RNN_NVN:
            self.init_rnn_type_nvn()
        elif self.net_type == NetType.DNN_MAXOUT:
            self.init_dnn_type_with_maxout()

    #init 文件里的配置
    def init_conf(self,conf):
        self.num_threads = conf.num_threads
        self.hidden_size = conf.hidden_size  # 隐藏层节点
        self.learning_rate = conf.learning_rate  # 学习速率
        self.num_layers = conf.num_layers  # 隐藏层数
        self.num_epochs = conf.num_epochs  # 训练周期
        self.exp_seq_len = conf.exp_seq_len  # 序列长度
        self.num_classes = conf.num_classes  # 分类个数
        self.num_features = conf.num_features  # 特征数量
        self.maxOut_numUnits = conf.maxOut_numUnits  # maxout节点
        self.embeded_dims = conf.embeded_dims  # 嵌入维数
        self.bias_initializer = tf.random_uniform_initializer(0, 0.001)  # bias初始器
        self.l2_preparam = conf.l2_preparam  # l2正则化超参数
        self.tensorboard  =conf.tensorboard
        self.use_tfrecord = conf.use_tfrecord
        self.tfrecord_path = conf.tfrecord_path
        self.shuffle = conf.shuffle
        self.keep_prob = conf.keep_prob
        self.use_dropout = conf.use_dropout

    #init 创建模型时的配置
    def init_config(self,config):
        # 将一些要创建时的数据通过config类传进来 包括模式，数据长度等等
        self.net_type = config.net_type  # 网络类型
        self.rnn_type = config.rnn_type  # rnn类型
        self.is_training = config.is_training  # 是否为训练模式
        self.is_test = config.is_test  # 是否为测试模式
        self.is_validation = config.is_validation  # 是否为验证模式
        self.len_features = config.len_features  # 特征长度
        self.train_seq_len = config.train_seq_len  # 训练集序列长度列表
        self.valid_seq_len = config.val_seq_len
        self.test_seq_len = config.test_seq_len
        self.activation = config.activation  # 激励函数
        self.batch_size = config.batch_size  # batch尺寸

    def init_rnn_type_nv1(self):

        # 输入数据
        self._input_data = tf.placeholder(tf.float32, [self.exp_seq_len, self.batch_size, self.len_features],
                                          name="input_data")
        self._targets = tf.placeholder(tf.int32,[self.batch_size],name="label")
        #self._valid_target = self._targets

        # 用于提前结束每个batch
        self._early_stop = tf.placeholder(tf.int32, shape=[self.batch_size], name="early-stop")

        if self.is_training:
            self.seq_len = self.exp_seq_len * self.batch_size
        elif self.is_validation:
            self.seq_len = sum(self.valid_seq_len)
        else:
            self.seq_len = sum(self.test_seq_len)

        # 获得多层双向gru的cell
        with tf.name_scope("mutil_rnn_cell"):
            cell = self.get_mutil_rnn_cell()

        # with tf.name_scope("embeded"):
        #   self.get_embeded_vec()

        # 初始化cell
        self.set_initial_states(cell)

        # 获得gru的输出
        with tf.name_scope("rnn_outputs"):
            self.get_rnn_outputs(cell)

            # softmax层的权重
        with tf.name_scope("softmax_layer") as scope:
            self.get_softmax_layer_output()

        # 获得混淆矩阵
        with tf.name_scope("confusion_matrix") as scope:
            self._confusion_matrix = tf.confusion_matrix(self._targets, self._digit_predictions,self.num_classes)

        with tf.name_scope("cross_entropy") as scope:
            self._onehot_labels = tf.one_hot(self._targets,depth=self.num_classes)
            self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._onehot_labels,logits=self._predictions)
            self._cross_entropy = tf.reduce_sum(self._loss)
            self._cost = tf.reduce_mean(self._loss)
            self.add_l2_regulation()

        with tf.name_scope("accuracy") as scope:
            self._correct_prediction = tf.equal(self._targets, self._digit_predictions)
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction,tf.float32))
            #self._accuracy = tf.metrics.accuracy( self._target,self._digit_predictions)[1]

        with tf.name_scope("optimization") as scope:
            self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cost,global_step=self.current_step)

        if self.tensorboard:
            self.w_hist = tf.summary.histogram("weights", self._softmax_w)
            self.b_hist = tf.summary.histogram("biases", self._softmax_b)
            self.y_hist_train = tf.summary.histogram("train-predictions", self._predictions)
            self.y_hist_test = tf.summary.histogram("test-predictions", self._predictions)
            self.mse_summary_train = tf.summary.scalar("train-cross-entropy-cost", self._cost)
            self.mse_summary_test = tf.summary.scalar("test-cross-entropy-cost", self._cost)

    def init_rnn_type_nvn(self):
        self._input_data = tf.placeholder(tf.float32, [self.exp_seq_len, self.batch_size, self.len_features],
                                          name="input_data")
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.exp_seq_len], name="targets")

        self._weight_sequence_loss = tf.placeholder(tf.float32,[self.batch_size,self.exp_seq_len],name="weight_sequence_loss")

        if self.is_training:
            self.seq_len = self.exp_seq_len * self.batch_size
        elif self.is_validation:
            self.seq_len = sum(self.valid_seq_len)
        else:
            self.seq_len = sum(self.test_seq_len)

        # 获得多层双向gru的cell
        with tf.name_scope("mutil_rnn_cell"):
            cell = self.get_mutil_rnn_cell()

        # 用于提前结束每个batch
        self._early_stop = tf.placeholder(tf.int32, shape=[self.batch_size], name="early-stop")

        # with tf.name_scope("embeded"):
        #   self.get_embeded_vec()

        # 初始化cell
        self.set_initial_states(cell)

        # 获得gru的输出
        with tf.name_scope("rnn_outputs"):
            self.get_rnn_outputs(cell)
        # 获得去除padding的标签
        # self._valid_target = self.get_valid_sequence(
        #     tf.reshape(self._targets, [self.exp_seq_len * self.batch_size]),
        #     self.num_classes)
        # softmax层的权重
        with tf.name_scope("softmax-layer") as scope:
            self.get_softmax_layer_output()

        # 获得混淆矩阵
        with tf.name_scope("confusion-matrix") as scope:
            self._confusion_matrix = tf.confusion_matrix(tf.reshape(self._targets,[self.batch_size*self.exp_seq_len]), self._digit_predictions,self.num_classes)

        with tf.name_scope("seq2seq-loss-by-example") as scpoe:
            self._loss = tf_ct.seq2seq.sequence_loss(tf.reshape(self._predictions,[self.batch_size,self.exp_seq_len,self.num_classes])
                                        ,self.targets,
                                        self._weight_sequence_loss)
            # self._loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            #     [self._predictions],
            #     [self._targets],
            #     [tf.ones([tf.cast(self.getTensorShape(self._targets)[0],tf.int32)])])

            self._cost = tf.reduce_mean(self._loss)
            self.add_l2_regulation()
            # 计算l2cost
            # tv = tf.trainable_variables()
            # #tf_ct.layers.l2_regularizer()
            #
            # self._regularization_cost = self.l2_preparam*tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            # #总cost为 基础cost + l2cost
            # #self._regularization_cost = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # self._cost = self._cost+self._regularization_cost
            self._accuracy = tf_ct.metrics.accuracy(self._digit_predictions, tf.reshape(self._targets,[self.batch_size*self.exp_seq_len]))

        with tf.name_scope("optimization") as scope:
            self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cost,global_step=self.current_step)

        if self.tensorboard:
            self.w_hist = tf.summary.histogram("weights", self._softmax_w)
            self.b_hist = tf.summary.histogram("biases", self._softmax_b)
            self.y_hist_train = tf.summary.histogram("train-predictions", self._predictions)
            self.y_hist_test = tf.summary.histogram("test-predictions", self._predictions)
            self.mse_summary_train = tf.summary.scalar("train-cross-entropy-cost", self._cost)
            self.mse_summary_test = tf.summary.scalar("test-cross-entropy-cost", self._cost)

    def init_dnn_type(self):
        # 输入数据
        self._input_data = tf.placeholder(tf.float32, [None, self.len_features],
                                          name="input_data")
        self._early_stop = tf.placeholder(tf.int32, shape=[None], name="early-stop")
        self._targets = tf.placeholder(tf.int32, [None], name="label")

        with tf.name_scope("init_weights") as scope:
            weights,biases = self.init_mutil_dnn_weights(self.len_features,self.hidden_size,self.num_classes,self.num_layers)

        with tf.name_scope("mlp") as scope:

            self._predictions = self.mlp(self._input_data,weights,biases,self.keep_prob)

        with tf.name_scope("cost") as scope:

            self._onehot_labels = tf.one_hot(self._targets, depth=self.num_classes)

            self._loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._onehot_labels,logits=self._predictions))
            self._cross_entropy = tf.reduce_sum(self._loss)
            self._cost = tf.reduce_mean(self._loss)

            self.add_l2_regulation()

        with tf.name_scope("accuracy") as scope:
            self._prob_predictions = tf.nn.softmax(self._predictions)
            # 获得每个数据最大的索引
            self._digit_predictions = tf.argmax(self._prob_predictions, axis=1, output_type=tf.int32)

            self._correct_prediction = tf.equal(self._targets, self._digit_predictions)
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction,tf.float32))
            #self._accuracy = tf.metrics.accuracy( self._valid_target,self._digit_predictions)[1]

        with tf.name_scope("confusion-matrix") as scope:
            self._confusion_matrix = tf.confusion_matrix(self._targets, self._digit_predictions,self.num_classes)

        with tf.name_scope("optimization") as scope:
            self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cost,global_step=self.current_step)

    def init_dnn_type_with_maxout(self):
        # 输入数据
        self._input_data = tf.placeholder(tf.float32, [None, self.len_features],
                                          name="input_data")
        self._early_stop = tf.placeholder(tf.int32, shape=[None], name="early-stop")
        self._targets = tf.placeholder(tf.int32, [None], name="label")

        with tf.name_scope("init_hidden_weights") as scope:
            weights,biases = self.init_mutil_dnn_weights(self.len_features,self.hidden_size,self.hidden_size,self.num_layers -1)

        with tf.name_scope("hidden_mlp") as scope:

            self._hidden_outputs = self.mlp(self._input_data,weights,biases,self.keep_prob)

        with tf.name_scope("maxout_layer") as scope:
            self._maxout_output = self.maxout(self._hidden_outputs,self.maxOut_numUnits,self.hidden_size,0,scope)


        with tf.name_scope("softmax_layer") as scope:
            self._softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.num_classes])

            self._softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.bias_initializer)

            self._predictions = tf.matmul(self._maxout_output,self._softmax_w) + self._softmax_b

            self._prob_predictions = tf.nn.softmax(self._predictions)

        with tf.name_scope("cost") as scope:

            self._onehot_labels = tf.one_hot(self._targets, depth=self.num_classes)

            self._cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._onehot_labels,logits=self._predictions))

            self.add_l2_regulation()

        with tf.name_scope("accuracy") as scope:
            # 获得每个数据最大的索引
            self._digit_predictions = tf.argmax(self._prob_predictions, axis=1, output_type=tf.int32)

            self._correct_prediction = tf.equal(self._targets, self._digit_predictions)
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction,tf.float32))
            #self._accuracy = tf.metrics.accuracy( self._valid_target,self._digit_predictions)[1]

        with tf.name_scope("confusion-matrix") as scope:
            self._confusion_matrix = tf.confusion_matrix(self._targets, self._digit_predictions,self.num_classes)

        with tf.name_scope("optimization") as scope:
            self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cost,global_step=self.current_step)


    def init_cnn_type(self):
        pass

    def init_mutil_dnn_weights(self,ils, hls, ols,hl_num):
        weights, bias = {}, {}
        stddev = 0.1
        for i in range(hl_num + 1):
            fan_in = ils if i == 0 else hls
            fan_out = ols if i == hl_num else hls
            weights[i] = tf.get_variable("weight_" + str(i),shape=[fan_in, fan_out])
            bias[i] = tf.get_variable("bias_"+str(i),shape = [fan_out])
        return weights, bias

    def mlp(self,_x, _w, _b, _keep_prob):
        layers = {}
        for i in range(len(_w)):
            if i == 0:
                if self.use_dropout:
                    layers[i] = tf.nn.dropout(self.activation(tf.add(tf.matmul(_x, _w[i]), _b[i])), _keep_prob)
                else:
                    layers[i] = self.activation(tf.add(tf.matmul(_x, _w[i]), _b[i]))
            elif i < len(_w) - 1:
                if self.use_dropout:
                    layers[i] = tf.nn.dropout(self.activation(tf.add(tf.matmul(layers[i - 1], _w[i]), _b[i])), _keep_prob)
                else:
                    layers[i] = self.activation(tf.add(tf.matmul(layers[i - 1], _w[i]), _b[i]))
            else:
                layers[i] = tf.add(tf.matmul(layers[i - 1], _w[i]), _b[i])
        return layers[len(_w) - 1]

    def get_embeded_vec(self):

        self._embeding_weights = tf.get_variable(name="embeding",shape=[self.len_features,self.embeded_dims],dtype=tf.float32)

        embed_input = tf.reshape(self._input_data,[self.exp_seq_len*self.batch_size,self.len_features])

        #embeding_bias = tf.get_variable(name="embeding_bias",shape=[self.embeded_dims],dtype=tf.float32,initializer=self.bias_initializer)

        embed_result = tf.matmul(embed_input,self.embeding_weights)# + embeding_bias

        self._embeded_result = tf.reshape(embed_result,[self.exp_seq_len,self.batch_size,self.embeded_dims])

    def get_mutil_rnn_cell(self):
        if self.use_dropout:
            if self.rnn_type == RNNType.GRU:
                cell = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(GRUCell(self.hidden_size, self.maxOut_numUnits, activation=self.activation),self.keep_prob,self.keep_prob,self.keep_prob) for _ in range(self.num_layers)])
                return cell
            elif self.rnn_type == RNNType.GRU_b:
                cell_fw = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(GRUCell(self.hidden_size, self.maxOut_numUnits, activation=self.activation),self.keep_prob,self.keep_prob,self.keep_prob) for _ in range(self.num_layers)])
                cell_bw = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(GRUCell(self.hidden_size, self.maxOut_numUnits, activation=self.activation),self.keep_prob,self.keep_prob,self.keep_prob) for _ in range(self.num_layers)])
                return (cell_fw,cell_bw)
            elif self.rnn_type == RNNType.LSTM:
                cell = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(BasicLSTMCell(self.hidden_size,activation=self.activation),self.keep_prob,self.keep_prob,self.keep_prob) for _ in range(self.num_layers)])
                return cell
            elif self.rnn_type == RNNType.LSTM_b:
                cell_fw = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(BasicLSTMCell(self.hidden_size,activation=self.activation),self.keep_prob,self.keep_prob,self.keep_prob) for _ in range(self.num_layers)])
                cell_bw = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(BasicLSTMCell(self.hidden_size,activation=self.activation),self.keep_prob,self.keep_prob,self.keep_prob) for _ in range(self.num_layers)])
                return (cell_fw,cell_bw)
            elif self.rnn_type == RNNType.NORM_GRU:
                cell = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(
                        BasicGRUCell(self.hidden_size, activation=self.activation), self.keep_prob,
                        self.keep_prob, self.keep_prob) for _ in range(self.num_layers)])
                return cell
            elif self.rnn_type == RNNType.NORM_GRU_b:
                cell_fw = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(BasicGRUCell(self.hidden_size, activation=self.activation),
                                              self.keep_prob, self.keep_prob, self.keep_prob) for _ in
                     range(self.num_layers)])
                cell_bw = tf_ct.rnn.MultiRNNCell(
                    [tf_ct.rnn.DropoutWrapper(BasicGRUCell(self.hidden_size, activation=self.activation),
                                              self.keep_prob, self.keep_prob, self.keep_prob) for _ in
                     range(self.num_layers)])
                return (cell_fw, cell_bw)
        else:

            if self.rnn_type == RNNType.GRU:
                cell = tf_ct.rnn.MultiRNNCell(
                    [GRUCell(self.hidden_size, self.maxOut_numUnits, activation=self.activation) for _ in range(self.num_layers)])
                return cell
            elif self.rnn_type == RNNType.GRU_b:
                cell_fw = tf_ct.rnn.MultiRNNCell(
                    [GRUCell(self.hidden_size, self.maxOut_numUnits, activation=self.activation) for _ in range(self.num_layers)])
                cell_bw = tf_ct.rnn.MultiRNNCell(
                    [GRUCell(self.hidden_size, self.maxOut_numUnits, activation=self.activation) for _ in range(self.num_layers)])
                return (cell_fw,cell_bw)
            elif self.rnn_type == RNNType.LSTM:
                cell = tf_ct.rnn.MultiRNNCell(
                    [BasicLSTMCell(self.hidden_size,activation=self.activation) for _ in range(self.num_layers)])
                return cell
            elif self.rnn_type == RNNType.LSTM_b:
                cell_fw = tf_ct.rnn.MultiRNNCell(
                    [BasicLSTMCell(self.hidden_size,activation=self.activation) for _ in range(self.num_layers)])
                cell_bw = tf_ct.rnn.MultiRNNCell(
                    [BasicLSTMCell(self.hidden_size,activation=self.activation) for _ in range(self.num_layers)])
                return (cell_fw,cell_bw)
            elif self.rnn_type == RNNType.NORM_GRU:
                cell = tf_ct.rnn.MultiRNNCell(
                    [BasicGRUCell(self.hidden_size, activation=self.activation) for _ in range(self.num_layers)])
                return cell
            elif self.rnn_type == RNNType.NORM_GRU_b:
                cell_fw = tf_ct.rnn.MultiRNNCell(
                    [BasicGRUCell(self.hidden_size, activation=self.activation) for _ in range(self.num_layers)])
                cell_bw = tf_ct.rnn.MultiRNNCell(
                    [BasicGRUCell(self.hidden_size, activation=self.activation) for _ in range(self.num_layers)])
                return (cell_fw, cell_bw)

    #初始化cell的状态
    def set_initial_states(self, cell):
        if self.rnn_type == RNNType.GRU or self.rnn_type == RNNType.LSTM or self.rnn_type == RNNType.NORM_GRU:
            self._initial_state = cell.zero_state(self.batch_size,tf.float32)
        elif self.rnn_type == RNNType.GRU_b or self.rnn_type == RNNType.LSTM_b or self.rnn_type == RNNType.NORM_GRU_b:
            (cell_fw, cell_bw) = cell
            self._initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            self._initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

    def get_rnn_outputs(self,cell):
        if self.rnn_type == RNNType.LSTM or self.rnn_type == RNNType.GRU or self.rnn_type == RNNType.NORM_GRU:
            self._outputs,self._state = tf.nn.dynamic_rnn(cell,self._input_data,sequence_length=self._early_stop,
                                              initial_state=self.initial_state,
                                              time_major=True,dtype=tf.float32)
            if self.net_type == NetType.RNN_NVN:
                outputs = tf.transpose(self._outputs, perm=[1, 0, 2])

                self._output = tf.reshape(tf.concat(axis=0, values=outputs),
                                         [self.exp_seq_len * self.batch_size, self.hidden_size])

                #self._valid_output = self.get_valid_sequence(self._output, self.hidden_size)

            elif self.net_type == NetType.RNN_NV1:
                if self.rnn_type == RNNType.LSTM :
                    state_h = self._state[-1]
                    self._output = state_h[-1]
                elif self.rnn_type == RNNType.GRU or self.rnn_type == RNNType.NORM_GRU:
                    self._output = self._state[-1]


        elif self.rnn_type == RNNType.LSTM_b or self.rnn_type == RNNType.GRU_b or self.rnn_type == RNNType.NORM_GRU_b:
            (cell_fw, cell_bw) = cell
            self._outputs, self._state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self._input_data,
                                                                       sequence_length=self._early_stop,
                                                                       initial_state_fw=self._initial_state_fw,
                                                                       initial_state_bw=self._initial_state_bw,
                                                                       time_major=True, dtype=tf.float32)

            if self.net_type == NetType.RNN_NVN:

                output_fw, output_bw = self._outputs
                output_fw = tf.transpose(output_fw, perm=[1, 0, 2])
                output_bw = tf.transpose(output_bw, perm=[1, 0, 2])
                outputs = tf.concat(axis=2, values=[output_fw, output_bw])
                # Concatenates tensors along one dimension.
                # this will flatten the dimension of the matrix to [batch_size * num_steps, num_hidden_nodes]
                # However, this is not the true output sequence, since padding added a number of empty elements
                # Extra padding elements should be removed from the output sequence.
                # Here first concatenate all vessels into one long sequence, including paddings
                self._output = tf.reshape(tf.concat(axis=0, values=outputs),
                                         [self.exp_seq_len * self.batch_size, self.hidden_size * 2])
                # Remove padding here
                #self._valid_output = self.get_valid_sequence(self._output, self.hidden_size * 2)
            elif self.net_type == NetType.RNN_NV1:
                state_fw, state_bw = self._state
                if self.rnn_type == RNNType.LSTM_b :
                    state_fw_h = state_fw[-1]
                    state_bw_h = state_bw[-1]
                    self._output = tf.concat(axis=1, values=[state_fw_h[-1], state_bw_h[-1]])
                elif self.rnn_type == RNNType.GRU_b or self.rnn_type==RNNType.NORM_GRU_b:
                    self._output = tf.concat(axis=1,values=[state_fw[-1],state_bw[-1]])

    def get_softmax_layer_output(self):
        if self.net_type == NetType.DNN:
            self._softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.num_classes])
        elif self.net_type == NetType.RNN_NV1 or self.net_type == NetType.RNN_NVN:

            if self.rnn_type == RNNType.GRU or self.rnn_type == RNNType.LSTM or self.rnn_type == RNNType.NORM_GRU:
                self._softmax_w = tf.get_variable("softmax_w", [self.hidden_size , self.num_classes])
            elif self.rnn_type == RNNType.GRU_b or self.rnn_type == RNNType.LSTM_b or self.rnn_type == RNNType.NORM_GRU_b:

                self._softmax_w = tf.get_variable("softmax_w", [self.hidden_size * 2, self.num_classes])

        # softmax层的bias
        self._softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.bias_initializer)

        self._predictions = tf.matmul(self._output, self._softmax_w) + self._softmax_b
        # 概率
        self._prob_predictions = tf.nn.softmax(self._predictions)
        # 获得每个数据最大的索引
        self._digit_predictions = tf.argmax(self._prob_predictions, axis=1,output_type=tf.int32)

    def add_l2_regulation(self):

        # 计算l2cost
        tv = tf.trainable_variables()

        # #tf_ct.layers.l2_regularizer()
        #
        self._l2_regularization_cost = self.l2_preparam*tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        # #总cost为 基础cost + l2cost
        self._cost = self._cost+self._l2_regularization_cost

    def maxout(self,input,num_units,output_size,ini_value,scope):
        outputs = None
        for i in range(num_units):
            with tf.variable_scope(str(i)):
                y = self.activation(linear._linear(input, output_size, True, ini_value, scope=scope))
                if outputs is None:
                    outputs = y
                else:
                    outputs = tf.maximum(outputs, y)
        return outputs

    def get_valid_sequence(self, seq, feature_size):
        """remove padding from sequences"""
        if self.is_training:
            stop = self._early_stop
        elif self.is_validation:
            stop = self._early_stop
        else:
            stop = self._early_stop
        valid_sequence_list = []
        for i in range(self.batch_size):
            if len(tf.Tensor.get_shape(seq)) == 2:
                sub_seq = tf.slice(seq, [self.exp_seq_len * i, 0], [stop[i], feature_size])
            else:
                sub_seq = tf.slice(seq, [self.exp_seq_len * i], [stop[i]])
            valid_sequence_list.append(sub_seq)
        valid_sequence = tf.concat(axis=0, values=valid_sequence_list)
        return valid_sequence

    def getTensorShape(this, tensor):
        return tf.Tensor.get_shape(tensor)

    @property
    def embeding_weights(self):
        return self._embeding_weights

    @property
    def embeded_result(self):
        return self._embeded_result

    @property
    def digit_predictions(self):
        return self._digit_predictions

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    @property
    def prob_predictions(self):
        return self._prob_predictions

    @property
    def input_data(self):
        return self._input_data

    @property
    def weight_sequence_loss(self):
        return self._weight_sequence_loss

    @property
    def targets(self):
        return self._targets

    @property
    def predictions(self):
        return self._predictions

    @property
    def early_stop(self):
        return self._early_stop

    @property
    def initial_state(self):
        if self.rnn_type == RNNType.GRU or self.rnn_type == RNNType.LSTM:
            return self._initial_state
        elif self.rnn_type == RNNType.GRU_b or self.rnn_type == RNNType.LSTM_b:
            return (self._initial_state_fw ,self._initial_state_bw)

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def train_op(self):
        return self._train_op

    @property
    def final_state(self):
        return self._final_state

