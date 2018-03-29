import numpy as np
import tensorflow as tf
import Data
import config
from model import Model
from tensorflow.python.ops.math_ops import tanh
import log


conf = config.Config("data/config.json")
log_path = "./log/"
data_path = "./data/"
task = conf.task
Log = log.Log(log_path,"training_data.csv")


def loadData():
    x_file = 'data/x_mobility.npy'
    y_file = 'data/y_mobility.npy'
    mmsi_file = 'data/mmsi_mobility.npy'


    #加载数据
    print("加载数据中......")
    x = np.load(x_file)
    y = np.load(y_file)
    mmsi = np.load(mmsi_file)
    print("加载完毕......")

    #x中数据格式如下
    # shape = [总序列个数，序列长度，特征数]
    #y中数据格式如下
    #shape= [总序列个数，序列长度]
    #mmsi数据格式如下
    #shape = [2，总序列个数]
    #mmsi[0] 中储存着用户编号
    #mmsi[1] 中储存着有效序列长度（因为是padding之后的切割，所以用户的一个序列会出现不满足序列长度的数据，故记录有效的序列长度，
    (x,y,mmsi) =  Data.Data.reorganizeSeq(x,y,mmsi,conf.exp_seq_len)

    #序列的总个数
    num_examples = x.shape[0]
    #用户编号的不重复列表
    unique_mmsi = np.unique(mmsi[0])
    #分类个数
    num_classes = conf.num_classes

    #测试集
    test_vessel = conf.test_id
    #验证集
    val_vessel = conf.val_id

    #分割数据，将数据分割成 训练集，测试集，验证集，返回这些数据集的索引
    #test_index 的格式
    #test_vessel = [0,1] 即前两名用户的索引 则test_index = [0,1,2,3,4,5,6......]
    (train_index, test_index, valid_index) = Data.Data.splitDataset(mmsi[0], test_vessel, val_vessel)

    #提前停止也即有效序列
    early_stop = mmsi[1]
    x = x.transpose([1, 0, 2])

    np.random.shuffle(train_index)


    # X_train shape = [序列长度，训练集序列总个数，特征]
    X_train = x[:, train_index, :]
    y_train = y[train_index, :]
    stop_train = early_stop[train_index]

    X_test = x[:, test_index, :]
    y_test = y[test_index, :]
    stop_test = early_stop[test_index]

    X_valid = x[:, valid_index, :]
    y_valid = y[valid_index, :]
    stop_valid = early_stop[valid_index]

    train_data = (X_train, y_train, stop_train)
    test_data = (X_test, y_test, stop_test)
    val_data = (X_valid, y_valid, stop_valid)



    #获得训练集，测试集，验证集的序列长度数组
    #eg train_seq_len  value = [250,250,250,55,250,250,250......]
    train_seq_len = mmsi[1][train_index]
    test_seq_len = mmsi[1][test_index]
    valid_seq_len = mmsi[1][valid_index]

    train_config = config.TrainingConfig(True, False,False, conf.batch_size)
    train_config.activation = tanh
    train_config.train_seq_len = train_seq_len
    train_config.len_features = x.shape[2]


    test_config = config.TrainingConfig(False,False,True,conf.batch_size)
    test_config.activation = tanh
    test_config.test_seq_len = test_seq_len
    test_config.len_features = x.shape[2]

    valid_config = config.TrainingConfig(False, True, False, conf.batch_size)
    valid_config.activation = tanh
    valid_config.val_seq_len = valid_seq_len
    valid_config.len_features = x.shape[2]

    return (train_data,test_data,val_data,train_config,test_config,valid_config)


def run_batch(session, m, data, eval_op, minibatch):
    # 准备数据
    x, y, e_stop = data
    epoch_size = x.shape[1] // m.batch_size

    # 记录结果
    costs = []
    correct = []

    for batch in range(epoch_size):
        x_batch = x[:, batch * m.batch_size: (batch + 1) * m.batch_size, :]
        y_batch = y[batch * m.batch_size: (batch + 1) * m.batch_size, :]
        e_batch = e_stop[batch * m.batch_size: (batch + 1) * m.batch_size]

        temp_dict = {m.input_data: x_batch}
        temp_dict.update({m.targets: y_batch})
        temp_dict.update({m.early_stop: e_batch})


        if m.is_training and eval_op == m.train_op:
            #如果是训练模式，且op正常 则正常训练
            print("开始训练第 %d 个batch" % batch)
            _, cost, accuracy = session.run([eval_op,m.cost, m.accuracy],
                                                               feed_dict=temp_dict)


        else:
            cost, confusion, accuracy, _ = session.run([m.cost, m.confusion_matrix, m._accuracy, eval_op],
                                                       feed_dict=temp_dict)

            # keep results for this minibatch
            costs.append(cost)
            correct.append(accuracy * sum(e_batch))

            # print test confusion matrix
            if not m.is_training and not m.is_validation:
                print('Confusion matrix on the test data:')
                print(confusion)
                # output predictions in test mode
                # if conf.test_mode:
                #     pred = session.run([m._prob_predictions], feed_dict=temp_dict)
                #     pred = np.array(pred)
                #     np.set_printoptions(threshold=np.nan)
                #     # results = np.column_stack((tar, pred))
                #     # np.savetxt("results/prediction.result", pred)#, fmt='%.3f')
                #     print("output target and predictions to file prediction.csv")
                #     exit()

            if batch == epoch_size - 1:
                accuracy = sum(correct) / float(sum(e_stop))
                return (sum(costs) / float(epoch_size), accuracy)

    # training: keep track of minibatch number
    return (minibatch)



def main():
    #1 处理数据
    #2 设置模型
    #3 训练模型
    #4 测试模型

    #x shape = [序列长度，总的序列个数，特征长度]
    #y shape = [总的序列个数，1}
    #early_stop  shape = [总的序列个数]  [250,250,250,250,50,.........]
    #train_index  训练集的索引 [10,11,12,13,......]
    train_data, test_data, val_data, train_config, test_config, valid_config = loadData()




    minibatch = 0

    with tf.Session() as sess:
        tf.set_random_seed(0)

        initializer = tf.random_uniform_initializer(0,0.001)

        with tf.variable_scope("model",reuse=False,initializer=initializer):
            train_model = Model(conf,train_config)
        # with tf.variable_scope("model",reuse=True,initializer=initializer):
        #     test_model = Model(conf,test_config)
        # with tf.variable_scope("model",reuse=True,initializer=initializer):
        #     valid_model = Model(conf,test_config)

        if conf.checkpoint or conf.restore:
            saver = tf.train.Saver()

        if conf.tensorboard:
            global writer
            writer = tf.summary.FileWriter(log_path, sess.graph_def)

        if not conf.restore:
            tf.global_variables_initializer().run()  # initialize all variables in the model
        else:
            saver.restore(sess, data_path + task)
            print("装载变量......")

        for i in range(conf.num_epochs):
            run_batch(sess,train_model,train_data,train_model.train_op,minibatch)




    pass



if __name__ == "__main__":

    main()