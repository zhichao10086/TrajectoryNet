import config
import tensorflow as tf
from customized_gru import CustomizedGRU as GRUCell
import log


class Model(object):

    def __init__(self,conf,config):
        self.num_threads = conf.num_threads
        self.hidden_size = conf.hidden_size
        self.learning_rate = conf.learning_rate
        self.num_layers = conf.num_layers
        self.num_epochs = conf.num_epochs
        self.exp_seq_len = conf.exp_seq_len
        self.num_classes = conf.num_classes
        self.num_features = conf.num_features
        self.maxOut_numUnits = conf.maxOut_numUnits
        self.embeded_dims = conf.embeded_dims
        self.bias_initializer = tf.random_uniform_initializer(0, 0.001)


        #将一些要创建时的数据通过config类传进来 包括模式，数据长度等等
        self.is_training = config.is_training       #是否为训练模式
        self.is_test = config.is_test               #是否为测试模式
        self.is_validation = config.is_validation   #是否为验证模式
        self.len_features = config.len_features     #特征长度
        self.train_seq_len = config.train_seq_len
        self.valid_seq_len = config.val_seq_len
        self.test_seq_len = config.test_seq_len
        self.activation = config.activation
        self.batch_size = config.batch_size

        self.current_step = tf.Variable(0)

        #输入数据
        self._input_data = tf.placeholder(tf.float32,[self.exp_seq_len,self.batch_size,self.len_features],name="input_data")

        self._targets = tf.placeholder(tf.int64,[self.batch_size,self.exp_seq_len],name="targets")

        if self.is_training:
            self.seq_len = self.exp_seq_len * self.batch_size
        elif self.is_validation:
            self.seq_len = self.valid_seq_len
        else:
            self.seq_len = self.test_seq_len

        #获得多层双向gru的cell
        with tf.name_scope("mutil_gru_cell") :
            cell = self.get_mutil_gru_cell()

        #用于提前结束每个batch
        self._early_stop = tf.placeholder(tf.int64, shape=[self.batch_size], name="early-stop")

        with tf.name_scope("embeded"):
            self.get_embeded_vec()

        #初始化cell
        self.set_initial_states(cell)

        #获得gru的输出
        with tf.name_scope("gru_outputs"):
            self.get_outputs(cell)

        #获得去除padding的标签
        self.valid_target = self.get_valid_sequence(tf.reshape(self._targets, [self.exp_seq_len * self.batch_size]),
                                                    self.num_classes)
        #softmax层的权重
        with tf.name_scope("softmax-W") as scope:
            softmax_w = self.get_softmax_layer()
            self.w = softmax_w

        #softmax层的bias
        with tf.name_scope("softmax-b") as scope:
            softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.bias_initializer)


        with tf.name_scope("softmax-predictions") as scope:

            self._predictions = tf.matmul(self.valid_output, softmax_w) + softmax_b
            #概率
            self._prob_predictions = tf.nn.softmax(self._predictions)
            #获得每个数据最大的索引
            self.digit_predictions = tf.argmax(self._prob_predictions, axis=1)


        #获得混淆矩阵
        with tf.name_scope("confusion-matrix") as scope:
            self.confusion_matrix = tf.confusion_matrix(self.valid_target, self.digit_predictions)


        with tf.name_scope("seq2seq-loss-by-example") as scpoe:
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self._predictions],
                [self.valid_target],
                [tf.ones([int(self.getTensorShape(self.valid_target)[0])])])
            self._cost = tf.reduce_mean(self.loss)
            self._accuracy = tf.contrib.metrics.accuracy(self.digit_predictions, self.valid_target)

        with tf.name_scope("optimization") as scope:
            self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self._cost,
                                                                                 global_step=self.current_step)


        if conf.tensorboard:
            self.w_hist = tf.summary.histogram("weights", softmax_w)
            self.b_hist = tf.summary.histogram("biases", softmax_b)
            self.y_hist_train = tf.summary.histogram("train-predictions", self._predictions)
            self.y_hist_test = tf.summary.histogram("test-predictions", self._predictions)
            self.mse_summary_train = tf.summary.scalar("train-cross-entropy-cost", self._cost)
            self.mse_summary_test = tf.summary.scalar("test-cross-entropy-cost", self._cost)



    def get_embeded_vec(self):

        self._embeding_weights = tf.get_variable(name="embeding",shape=[self.len_features,self.embeded_dims],dtype=tf.float32)

        embed_input = tf.reshape(self._input_data,[self.exp_seq_len*self.batch_size,self.len_features])

        #embeding_bias = tf.get_variable(name="embeding_bias",shape=[self.embeded_dims],dtype=tf.float32,initializer=self.bias_initializer)

        embed_result = tf.matmul(embed_input,self.embeding_weights)# + embeding_bias

        self._embeded_result = tf.reshape(embed_result,[self.exp_seq_len,self.batch_size,self.embeded_dims])



    #获取多层gru
    def get_mutil_gru_cell(self):
        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [GRUCell(self.hidden_size,self.maxOut_numUnits, activation=self.activation) for _ in range(self.num_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [GRUCell(self.hidden_size,self.maxOut_numUnits, activation=self.activation) for _ in range(self.num_layers)])
        return (cell_fw, cell_bw)


    #初始化cell的状态
    def set_initial_states(self, cell):
        (cell_fw, cell_bw) = cell
        self.initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        self.initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)


    def get_outputs(self,cell):

        (cell_fw, cell_bw) = cell
        self.outputs, self.state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embeded_result,
                                                                   sequence_length=self._early_stop,
                                                                   initial_state_fw=self.initial_state_fw,
                                                                   initial_state_bw=self.initial_state_bw,
                                                                   time_major=True, dtype=tf.float32)
        output_fw, output_bw = self.outputs
        output_fw = tf.transpose(output_fw, perm=[1, 0, 2])
        output_bw = tf.transpose(output_bw, perm=[1, 0, 2])
        outputs = tf.concat(axis=2, values=[output_fw, output_bw])
        # Concatenates tensors along one dimension.
        # this will flatten the dimension of the matrix to [batch_size * num_steps, num_hidden_nodes]
        # However, this is not the true output sequence, since padding added a number of empty elements
        # Extra padding elements should be removed from the output sequence.
        # Here first concatenate all vessels into one long sequence, including paddings
        self.output = tf.reshape(tf.concat(axis=0, values=outputs),
                                 [self.exp_seq_len * self.batch_size, self.hidden_size * 2])
        # Remove padding here
        self.valid_output = self.get_valid_sequence(self.output, self.hidden_size * 2)

    def get_softmax_layer(self):
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size * 2, self.num_classes])

        return softmax_w


    def get_valid_sequence(self, seq, feature_size):
        """remove padding from sequences"""
        if self.is_training:
            stop = self.train_seq_len
        elif self.is_validation:
            stop = self.valid_seq_len
        else:
            stop = self.test_seq_len
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
    def prob_predictions(self):
        return self._prob_predictions

    @property
    def input_data(self):
        return self._input_data

    @property
    def inputs(self):
        return self._inputs

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
        return self._initial_state

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

