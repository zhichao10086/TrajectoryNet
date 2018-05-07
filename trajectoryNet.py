import numpy as np
import tensorflow as tf
import pandas as pd
import config
from config import TrainingConfig
from model import Model
from config import NetType
from config import RNNType
from tensorflow.python.ops.math_ops import tanh
from log import Log
from myThread import MyThread
import random
from tensorflow.contrib import layers
from data_funs import Data
import util
import param
from param import DirName
from sklearn.metrics import confusion_matrix
from util import get_net_type
from util import get_rnn_type


conf = config.Config("data/config.json")

if conf.use_tfrecord:
    log_path,data_path,LOGGER = util.init_environment(get_net_type(conf.net_type),get_rnn_type(conf.rnn_type))
    task = conf.task
    tfrecords_data_path = conf.tfrecord_path
    len_features = conf.num_features * conf.discretization_width
else:
    log_path = "./logdir/shiyan/rnn_nvn/"
    data_path = "./data/tfrecord9_data/rnn_nvn/"
    task = conf.task
    net_name = str(NetType(conf.net_type)).split(".")[-1]
    nn_name = str(RNNType(conf.rnn_type)).split(".")[-1]
    LOGGER = Log(log_path, "_" + net_name + "_" + nn_name)
    len_features = conf.num_features * conf.discretization_width

#从npy加载数据
def loadData_rnn_nvn():
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
    (x,y,mmsi) =  Data.reorganizeSeq(x, y, mmsi, conf.exp_seq_len)

    #序列的总个数
    num_examples = x.shape[0]
    #用户编号的不重复列表
    unique_mmsi = np.unique(mmsi[0])
    #分类个数
    num_classes = conf.num_classes

    #test_and_val = random.sample(range(23),6)

    #测试集
    #test_vessel = test_and_val[0:5]
    test_vessel = conf.test_id
    #验证集
    #val_vessel = test_and_val[5:6]
    val_vessel = conf.val_id

    #分割数据，将数据分割成 训练集，测试集，验证集，返回这些数据集的索引
    #test_index 的格式
    #test_vessel = [0,1] 即前两名用户的索引 则test_index = [0,1,2,3,4,5,6......]
    (train_index, test_index, valid_index) = Data.splitDataset(mmsi[0], test_vessel, val_vessel)

    #提前停止也即有效序列
    early_stop = mmsi[1]
    x = x.transpose([1, 0, 2])

    np.random.shuffle(train_index)


    # X_train shape = [序列长度，训练集序列总个数，特征]
    X_train = x[:, train_index, :]
    y_train = y[train_index, :]
    stop_train = early_stop[train_index]

    np.random.shuffle(test_index)

    X_test = x[:, test_index, :]
    y_test = y[test_index, :]
    stop_test = early_stop[test_index]

    np.random.shuffle(valid_index)

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

    train_config = config.TrainingConfig(True, False,False, conf.batch_size,len_features=x.shape[2],rnn_type=RNNType.GRU_b)
    train_config.train_seq_len = train_seq_len

    test_config = config.TrainingConfig(False,False,True,len(test_index),len_features=x.shape[2],rnn_type=RNNType.GRU_b)
    test_config.test_seq_len = test_seq_len

    valid_config = config.TrainingConfig(False, True, False, len(valid_index),len_features=x.shape[2],rnn_type=RNNType.GRU_b)
    valid_config.val_seq_len = valid_seq_len

    return (train_data,test_data,val_data,train_config,test_config,valid_config)

#加载rnn_nv1数据从npy文件
def load_data_rnn_nv1(classes):
    # 分训练集与测试集 验证集 8：1：1
    train_data_all = None
    train_label_all = None
    train_early_all = None
    valid_data_all = None
    valid_label_all = None
    valid_early_all = None
    test_data_all = None
    test_label_all = None
    test_early_all = None
    features_arr_list = []
    index_arr_list = []
    label_arr_list = []
    data_file_name_exp = data_path +"transportation_mode"
    for i in range(classes):
        print("加载" + str(i))
        # data_file  = data_file_name +str(i) +".npy"
        index_df = pd.DataFrame(pd.read_csv(data_file_name_exp +"_"+ str(i) + "_seg_index.csv"))
        features_arr = np.load(data_file_name_exp + str(i) + ".npy")
        print(features_arr.shape)
        features_arr = features_arr[:, 0:len_features]
        index_arr = np.array(index_df.iloc[:, [1, 2]].T)
        # index shape = [2,总个数]
        # 第一维是第几段轨迹 第二维是在固定长度为exp_seq_len中的实际长度
        # data shape =[seq_nums,exp_seq_len,feature_len]   切出相等的数据长度 不足的padding
        (data, index_arr) = Data.slice_seq(features_arr, index_arr, conf.exp_seq_len)
        #切割后删除features_arr index
        del features_arr
        del index_df
        label_arr = np.zeros(shape=[index_arr.shape[1]], dtype=np.int32)
        label_arr[:] = i
        # features_arr_list.append(data)
        # index_arr_list.append(index)
        # label_arr_list.append(label)
        #划分训练集，验证集，测试集
        print("划分训练集，验证集，测试集   " + str(i))
        seq_nums = index_arr.shape[1]
        # 控制变量
        np.random.seed(2)
        index_perm = np.random.permutation(range(seq_nums))
        train_count = int(np.floor(seq_nums * 0.8))
        valid_count = int(np.floor(seq_nums * 0.9))
        test_count = seq_nums
        train_index = index_perm[0:train_count]
        valid_index = index_perm[train_count + 1:valid_count]
        test_index = index_perm[valid_count + 1:seq_nums]

        # train_set valid_set test_set
        train_data = data[train_index, :, :]
        train_label = label_arr[train_index]
        train_early = index_arr[1, train_index]

        valid_data = data[valid_index, :, :]
        valid_label = label_arr[valid_index]
        valid_early = index_arr[1, valid_index]

        test_data = data[test_index, :, :]
        test_label = label_arr[test_index]
        test_early = index_arr[1, test_index]

        #删除读取到的data.
        del data
        del label_arr
        del index_arr

        if train_data_all is None:
            train_data_all = train_data
            train_label_all = train_label
            train_early_all = train_early

            valid_data_all = valid_data
            valid_label_all = valid_label
            valid_early_all = valid_early

            test_data_all = test_data
            test_label_all = test_label
            test_early_all = test_early
        else:
            train_data_all = np.concatenate((train_data_all, train_data), axis=0)
            train_label_all = np.concatenate((train_label_all, train_label), axis=0)
            train_early_all = np.concatenate((train_early_all, train_early), axis=0)

            valid_data_all = np.concatenate((valid_data_all, valid_data), axis=0)
            valid_label_all = np.concatenate((valid_label_all, valid_label), axis=0)
            valid_early_all = np.concatenate((valid_early_all, valid_early), axis=0)

            test_data_all = np.concatenate((test_data_all, test_data), axis=0)
            test_label_all = np.concatenate((test_label_all, test_label), axis=0)
            test_early_all = np.concatenate((test_early_all, test_early), axis=0)
    #打乱数据
    np.random.seed(1)
    train_perm = np.random.permutation(range(train_early_all.shape[0]))
    np.random.seed(1)
    valid_perm = np.random.permutation(range(valid_early_all.shape[0]))
    np.random.seed(1)
    test_perm = np.random.permutation(range(test_early_all.shape[0]))

    #shape=[序列长度，总个数，特征长度]   TimeMajor
    train_data_all = np.transpose(train_data_all, [1, 0, 2])
    valid_data_all = np.transpose(valid_data_all, [1, 0, 2])
    test_data_all = np.transpose(test_data_all, [1, 0, 2])

    # train_data_all = train_data_all[:, train_perm, :]
    # train_label_all = train_label_all[train_perm]
    # train_early_all = train_early_all[train_perm]

    valid_data_all = valid_data_all[:, valid_perm, :]
    valid_label_all = valid_label_all[valid_perm]
    valid_early_all = valid_early_all[valid_perm]

    test_data_all = test_data_all[:, test_perm, :]
    test_label_all = test_label_all[test_perm]
    test_early_all = test_early_all[test_perm]

    train_set = (train_data_all, train_label_all, train_early_all)
    valid_set = (valid_data_all, valid_label_all, valid_early_all)
    test_set = (test_data_all, test_label_all, test_early_all)
    return train_set,valid_set,test_set

def load_data_rnn_nv1_other(classes):
    # 分训练集与测试集 验证集 8：1：1
    train_data_all = None
    train_label_all = None
    train_early_all = None
    valid_data_all = None
    valid_label_all = None
    valid_early_all = None
    test_data_all = None
    test_label_all = None
    test_early_all = None
    data_file_name_exp = data_path + "transportation_mode"
    for i in range(classes):
        data = np.load(data_path + "slice_label" + str(i) + "_" + str(conf.exp_seq_len) + ".npy")
        index_arr = np.load(data_path + "slice_index" + str(i) + ".npy")

        # 切割后删除features_arr index
        label_arr = np.zeros(shape=[index_arr.shape[1]], dtype=np.int32)
        label_arr[:] = i
        # features_arr_list.append(data)
        # index_arr_list.append(index)
        # label_arr_list.append(label)
        # 划分训练集，验证集，测试集
        print("划分训练集，验证集，测试集   " + str(i))
        seq_nums = index_arr.shape[1]
        # 控制变量
        np.random.seed(2)
        index_perm = np.random.permutation(range(seq_nums))
        train_count = int(np.floor(seq_nums * 0.8))
        valid_count = int(np.floor(seq_nums * 0.9))
        test_count = seq_nums
        train_index = index_perm[0:train_count]
        valid_index = index_perm[train_count + 1:valid_count]
        test_index = index_perm[valid_count + 1:seq_nums]

        # train_set valid_set test_set
        train_data = data[train_index, :, :]
        train_label = label_arr[train_index]
        train_early = index_arr[1, train_index]

        valid_data = data[valid_index, :, :]
        valid_label = label_arr[valid_index]
        valid_early = index_arr[1, valid_index]

        test_data = data[test_index, :, :]
        test_label = label_arr[test_index]
        test_early = index_arr[1, test_index]

        # 删除读取到的data.
        del data
        del label_arr
        del index_arr

        if train_data_all is None:
            train_data_all = train_data
            train_label_all = train_label
            train_early_all = train_early

            valid_data_all = valid_data
            valid_label_all = valid_label
            valid_early_all = valid_early

            test_data_all = test_data
            test_label_all = test_label
            test_early_all = test_early
        else:
            train_data_all = np.concatenate((train_data_all, train_data), axis=0)
            train_label_all = np.concatenate((train_label_all, train_label), axis=0)
            train_early_all = np.concatenate((train_early_all, train_early), axis=0)

            valid_data_all = np.concatenate((valid_data_all, valid_data), axis=0)
            valid_label_all = np.concatenate((valid_label_all, valid_label), axis=0)
            valid_early_all = np.concatenate((valid_early_all, valid_early), axis=0)

            test_data_all = np.concatenate((test_data_all, test_data), axis=0)
            test_label_all = np.concatenate((test_label_all, test_label), axis=0)
            test_early_all = np.concatenate((test_early_all, test_early), axis=0)
    # 打乱数据
    np.random.seed(1)
    train_perm = np.random.permutation(range(train_early_all.shape[0]))
    np.random.seed(1)
    valid_perm = np.random.permutation(range(valid_early_all.shape[0]))
    np.random.seed(1)
    test_perm = np.random.permutation(range(test_early_all.shape[0]))

    # shape=[序列长度，总个数，特征长度]   TimeMajor
    train_data_all = np.transpose(train_data_all, [1, 0, 2])
    valid_data_all = np.transpose(valid_data_all, [1, 0, 2])
    test_data_all = np.transpose(test_data_all, [1, 0, 2])

    # train_data_all = train_data_all[:, train_perm, :]
    # train_label_all = train_label_all[train_perm]
    # train_early_all = train_early_all[train_perm]

    valid_data_all = valid_data_all[:, valid_perm, :]
    valid_label_all = valid_label_all[valid_perm]
    valid_early_all = valid_early_all[valid_perm]

    test_data_all = test_data_all[:, test_perm, :]
    test_label_all = test_label_all[test_perm]
    test_early_all = test_early_all[test_perm]

    train_set = (train_data_all, train_label_all, train_early_all)
    valid_set = (valid_data_all, valid_label_all, valid_early_all)
    test_set = (test_data_all, test_label_all, test_early_all)
    return train_set, valid_set, test_set

#直接读取整个data npz  noTranspose
def load_data_rnn_nv1_quick(classes):
    data_dir = "G:/all_data/"
    train_data_set_name = data_dir + "train_data_set.npz"
    valid_data_set_name = data_dir + "valid_data_set.npz"
    test_data_set_name = data_dir + "test_data_set.npz"
    train_data_set = np.load(train_data_set_name)
    valid_data_set = np.load(valid_data_set_name)
    test_data_set = np.load(test_data_set_name)

    return train_data_set,valid_data_set,test_data_set

def evaluate_model(sess, minibatch):
    # test and validate model
    #if conf.test_mode:
    #    run_batch(sess, mtest, test_data, tf.no_op(), minibatch)

    result_train = run_batch(sess,train_model,train_data,tf.no_op(),minibatch)
    result_test = run_batch(sess,test_model,test_data,tf.no_op(),minibatch)
    result_valid = run_batch(sess,valid_model,valid_data,tf.no_op(),minibatch)

    #t_train = MyThread(run_batch, (sess, train_model, train_data, tf.no_op(), minibatch))
    #t_test = MyThread(run_batch, (sess, test_model, test_data, tf.no_op(), minibatch))
    #t_val = MyThread(run_batch, (sess, valid_model, valid_data, tf.no_op(), minibatch))

    #t_train.start()
    #t_test.start()
    #t_val.start()

    #t_train.join()
    #result_train = t_train.get_result()
    #t_test.join()
    #result_test = t_test.get_result()
    #t_val.join()
    #result_val = t_val.get_result()

    print("Train cost {0:0.3f}, Acc {1:0.3f}".format(
        result_train[0], result_train[1]))
    print("Valid cost {0:0.3f}, Acc {1:0.3f}".format(
        result_valid[0], result_valid[1]))
    print("Test  cost {0:0.3f}, Acc {1:0.3f}".format(
        result_test[0], result_test[1]))

    return result_train + result_test + result_valid

#
def evaluate_model_all(sess,epoch):
    result_train = run_batch_all(sess, train_model, train_data, tf.no_op(), epoch)
    result_valid = run_batch_all(sess, valid_model, valid_data, tf.no_op(), epoch)
    result_test = run_batch_all(sess, test_model, test_data, tf.no_op(), epoch)


    LOGGER.summary_log(result_train+result_valid+result_test,epoch)

    print("Train cost {0:0.3f}, Acc {1:0.3f}".format(
        result_train[0], result_train[1]))
    print("Valid cost {0:0.3f}, Acc {1:0.3f}".format(
        result_valid[0], result_valid[1]))
    print("Test  cost {0:0.3f}, Acc {1:0.3f}".format(
        result_test[0], result_test[1]))

    return result_train + result_test + result_valid

#npz文件方式
def evaluate_model_quick(sess,epoch):
    print("开始测试训练集")
    result_train = run_batch_quick(sess, train_model, train_data, tf.no_op(), epoch)
    print("开始测试验证集")
    result_valid = run_batch_quick(sess, valid_model, valid_data, tf.no_op(), epoch)
    print("开始测试测试集")
    result_test = run_batch_quick(sess, test_model, test_data, tf.no_op(), epoch)

    LOGGER.summary_log(result_train + result_valid + result_test, epoch)

    print("Train cost {0:0.3f}, Acc {1:0.3f}".format(
        result_train[0], result_train[1]))
    print("Valid cost {0:0.3f}, Acc {1:0.3f}".format(
        result_valid[0], result_valid[1]))
    print("Test  cost {0:0.3f}, Acc {1:0.3f}".format(
        result_test[0], result_test[1]))

    return result_train + result_test + result_valid

#队列版 未完成
def evaluate_from_tfrecords(iter):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(len(threads))
        train_cost, train_acc = run_batch_from_tfrecords(sess, coord, train_model, tf.no_op())

        coord.request_stop()
        coord.join(threads)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        valid_cost, valid_acc = run_batch_from_tfrecords(sess, coord, valid_model, tf.no_op())

        coord.request_stop()
        coord.join(threads)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        test_cost, test_acc = run_batch_from_tfrecords(sess, coord, test_model, tf.no_op())

        coord.request_stop()
        coord.join(threads)

    LOGGER.summary_log((train_cost, train_acc, valid_cost, valid_acc, test_cost, test_acc),iter)

    print("Train cost {0:0.3f}, Acc {1:0.3f}".format(
        train_cost, train_acc))
    print("Valid cost {0:0.3f}, Acc {1:0.3f}".format(
        valid_cost, valid_acc))
    print("Test  cost {0:0.3f}, Acc {1:0.3f}".format(
        test_cost, test_acc))

#dataset版
def evaluate_from_tfrecord_dataset(net_type,sess, model,next_element,eval_op,epoch):

    cost_list = []
    acc_list = []
    confus_list = []
    count = 0
    try:
        while True:
            input, early, label = sess.run(next_element)

            if net_type == NetType.RNN_NV1:

                if input.shape[0] < conf.batch_size:
                    print(input.shape)
                    break
                input = np.transpose(input, [1, 0, 2])
                batch_size = input.shape[1]

                cost, acc, confus_mat = sess.run(fetches=[model.cost, model.accuracy, model.confusion_matrix],
                                                 feed_dict={model.input_data: input,
                                                            model.early_stop: early,
                                                            model.targets: label})

            elif net_type == NetType.RNN_NVN:
                if input.shape[0] < conf.batch_size:
                    print(input.shape)
                    break
                new_label = np.zeros([conf.batch_size, conf.exp_seq_len], np.int32)
                for batch in range(conf.batch_size):
                    new_label[batch, 0:early[batch]] = label[batch]
                    new_label[batch, early[batch]:] = 0
                label = new_label
                input = np.transpose(input, [1, 0, 2])
                batch_size = input.shape[1]

                weight_sequence_loss = np.zeros([conf.batch_size, conf.exp_seq_len], np.float32)
                for k in range(conf.batch_size):
                    weight_sequence_loss[k, 0:early[k]] = 1

                cost,digit_predictions = sess.run(fetches = [model.cost,model.digit_predictions],feed_dict={
                                                                model.input_data:input,
                                                                model.early_stop:early,
                                                                model.weight_sequence_loss:weight_sequence_loss,
                                                                model.targets:label
                                                            })


                batch_acc_list = []
                confus_mat_list = []
                for k in range(conf.batch_size):
                    start = k*conf.exp_seq_len
                    end = k*conf.exp_seq_len + early[k]
                    seq_acc = np.equal(digit_predictions[start:end],label[k,0:early[k]])
                    seq_acc = seq_acc.astype(np.float32)
                    batch_acc_list.append(np.mean(seq_acc))
                    confus = confusion_matrix(label[k,0:early[k]],digit_predictions[start:end],labels = [0,1,2,3])
                    confus_mat_list.append(confus)
                acc = sum(batch_acc_list)/conf.batch_size
                confus_mat = sum(confus_mat_list)
            elif net_type == NetType.DNN or net_type == NetType.DNN_MAXOUT:
                list_input = []
                list_label = []
                for batch in range(input.shape[0]):
                    list_input.append(input[batch, 0:early[batch], :])
                    new_label = np.zeros([early[batch]], np.int32)
                    new_label[:] = label[batch]
                    list_label.append(new_label)
                input = np.concatenate(tuple(list_input), axis=0)
                label = np.concatenate(tuple(list_label), axis=0)
                batch_size = input.shape[0]
                cost, acc, confus_mat = sess.run(fetches=[model.cost, model.accuracy, model.confusion_matrix],
                                                 feed_dict={model.input_data: input,
                                                            model.early_stop: early,
                                                            model.targets: label})

            #print(input.shape)

            confus_list.append(confus_mat)
            cost_list.append(cost)
            acc_list.append(acc)
            count += 1

    except tf.errors.OutOfRangeError:

        print("超出界限！！！")

    if model.is_training:
        LOGGER.training_log("训练集：\n")
    elif model.is_validation:
        LOGGER.training_log("验证集：\n")
    else:
        LOGGER.training_log("测试集：\n")
    LOGGER.training_log(str(sum(confus_list)))
    print( count)
    return sum(cost_list) / len(cost_list), sum(acc_list) / len(acc_list)

#minbatch训练方法
def run_batch(session, model, data, eval_op, minibatch):
    # 准备数据
    x, y, e_stop = data
    epoch_size = x.shape[1] // model.batch_size

    # 记录结果
    costs = []
    correct = []

    for batch in range(epoch_size):
        x_batch = x[:, batch * model.batch_size: (batch + 1) * model.batch_size, :]
        y_batch = y[batch * model.batch_size: (batch + 1) * model.batch_size]
        e_batch = e_stop[batch * model.batch_size: (batch + 1) * model.batch_size]

        temp_dict = {model.input_data: x_batch}
        temp_dict.update({model.targets: y_batch})
        temp_dict.update({model.early_stop: e_batch})


        if model.is_training and eval_op == model.train_op:
            #如果是训练模式，且op正常 则正常训练
            print("开始训练第 %d 个batch" % batch)
            _, cost, accuracy = session.run([eval_op, model.cost, model.accuracy],
                                            feed_dict=temp_dict)

            if minibatch % conf.evaluate_freq == 0:
                result = evaluate_model(session, minibatch)  #评估模型，返回结果
                LOGGER.summary_log(result, minibatch)
            minibatch += 1


        else:
            cost, confusion, accuracy, _ = session.run([model.cost, model.confusion_matrix, model.accuracy, eval_op],
                                                       feed_dict=temp_dict)

            if model.net_type == NetType.RNN_NVN:
                # keep results for this minibatch
                costs.append(cost)
                correct.append(accuracy * sum(e_batch))

                # print test confusion matrix
                if not model.is_training and not model.is_validation:

                    LOGGER.training_log(str(minibatch) + "测试集的混淆矩阵")
                    LOGGER.training_log(str(confusion))
                    # output predictions in test mode
                    # if conf.test_mode:
                    #     pred = session.run([m._prob_predictions], feed_dict=temp_dict)
                    #     pred = np.array(pred)
                    #     np.set_printoptions(threshold=np.nan)
                    #     # results = np.column_stack((tar, pred))
                    #     # np.savetxt("results/prediction.result", pred)#, fmt='%.3f')
                    #     #print("output target and predictions to file prediction.csv")
                    #     #exit()

                #计算平均精度与损失
                if batch == epoch_size - 1:
                    accuracy = sum(correct) / float(sum(e_stop))
                    return (sum(costs) / float(epoch_size), accuracy)
            elif model.net_type == NetType.RNN_NV1:
                costs.append(cost)
                correct.append(accuracy)

                # print test confusion matrix
                if not model.is_training and not model.is_validation:
                    LOGGER.training_log(str(minibatch) + "测试集的混淆矩阵")
                    LOGGER.training_log(str(confusion))
                    # output predictions in test mode
                    # if conf.test_mode:
                    #     pred = session.run([m._prob_predictions], feed_dict=temp_dict)
                    #     pred = np.array(pred)
                    #     np.set_printoptions(threshold=np.nan)
                    #     # results = np.column_stack((tar, pred))
                    #     # np.savetxt("results/prediction.result", pred)#, fmt='%.3f')
                    #     #print("output target and predictions to file prediction.csv")
                    #     #exit()

                # 计算平均精度与损失
                if batch == epoch_size - 1:
                    cost_mean = (sum(costs) )/ float(epoch_size)
                    accuracy_mean = sum(correct) / float(epoch_size)
                    return (cost_mean,accuracy_mean)

    # training: keep track of minibatch number
    return (minibatch)

def run_batch_all(session,model,data,eval_op,epoch):
    x, y, e_stop = data
    epoch_size = x.shape[1] // model.batch_size
    shuffle_perm = None
    if model.is_training:
        shuffle_perm = np.random.permutation(range(e_stop.shape[0]))


    # 记录结果
    costs = []
    correct = []
    for batch in range(epoch_size):

        if model.is_training:
            batch_perm = shuffle_perm[batch * model.batch_size: (batch + 1) * model.batch_size]
            x_batch = x[:, batch_perm, :]
            y_batch = y[batch_perm]
            e_batch = e_stop[batch_perm]
        else:
            x_batch = x[:, batch * model.batch_size: (batch + 1) * model.batch_size, :]
            y_batch = y[batch * model.batch_size: (batch + 1) * model.batch_size]
            e_batch = e_stop[batch * model.batch_size: (batch + 1) * model.batch_size]

        temp_dict = {model.input_data: x_batch}
        temp_dict.update({model.targets: y_batch})
        temp_dict.update({model.early_stop: e_batch})

        if model.is_training and eval_op == model.train_op:
            _= session.run([eval_op],feed_dict=temp_dict)

        else:
            cost, confusion, accuracy, _ = session.run([model.cost, model.confusion_matrix, model.accuracy, eval_op],feed_dict=temp_dict)

            if model.is_test:
                LOGGER.training_log(str(epoch) + "测试集的混淆矩阵")
                LOGGER.training_log(str(confusion))
            elif model.is_validation:
                LOGGER.training_log(str(epoch) + "验证集的混淆矩阵")
                LOGGER.training_log(str(confusion))


            if model.net_type == NetType.RNN_NVN:
                # keep results for this minibatch
                costs.append(cost)
                correct.append(accuracy * sum(e_batch))

                #计算平均精度与损失
                if batch == epoch_size - 1:
                    accuracy = sum(correct) / float(sum(e_stop))
                    return (sum(costs) / float(epoch_size), accuracy)
            elif model.net_type == NetType.RNN_NV1:
                costs.append(cost)
                correct.append(accuracy)

                # 计算平均精度与损失
                if batch == epoch_size - 1:
                    cost_mean = sum(costs) / float(epoch_size)
                    accuracy_mean = sum(correct) / float(epoch_size)
                    return (cost_mean, accuracy_mean)

#quick 代表所有数据在npz文件里
def run_batch_quick(session,model,data,eval_op,epoch):

    if model.is_training:
        x = data["train_data"]
        y = data["train_label"]
        e_stop = data["train_early_stop"]
    elif model.is_validation:
        x = data["valid_data"]
        y = data["valid_label"]
        e_stop = data["valid_early_stop"]
    else:
        x = data["test_data"]
        y = data["test_label"]
        e_stop = data["test_early_stop"]

    epoch_size = x.shape[0] // model.batch_size
    shuffle_perm = None
    if model.is_training:
        shuffle_perm = np.random.permutation(range(e_stop.shape[0]))

    # 记录结果
    costs = []
    correct = []
    for batch in range(epoch_size):

        if model.is_training:
            batch_perm = shuffle_perm[batch * model.batch_size: (batch + 1) * model.batch_size]
            x_batch = x[batch_perm,:, :]
            y_batch = y[batch_perm]
            e_batch = e_stop[batch_perm]
        else:
            x_batch = x[batch * model.batch_size: (batch + 1) * model.batch_size,: , :]
            y_batch = y[batch * model.batch_size: (batch + 1) * model.batch_size]
            e_batch = e_stop[batch * model.batch_size: (batch + 1) * model.batch_size]

        x_batch = np.transpose(x_batch,[1,0,2])

        temp_dict = {model.input_data: x_batch}
        temp_dict.update({model.targets: y_batch})
        temp_dict.update({model.early_stop: e_batch})

        if model.is_training and eval_op == model.train_op:
            _ = session.run([eval_op], feed_dict=temp_dict)

        else:
            cost, confusion, accuracy, _ = session.run([model.cost, model.confusion_matrix, model.accuracy, eval_op],
                                                       feed_dict=temp_dict)

            if model.is_test:
                LOGGER.training_log(str(epoch) + "测试集的混淆矩阵")
                LOGGER.training_log(str(confusion))
            elif model.is_validation:
                LOGGER.training_log(str(epoch) + "验证集的混淆矩阵")
                LOGGER.training_log(str(confusion))

            if model.net_type == NetType.RNN_NVN:
                # keep results for this minibatch
                costs.append(cost)
                correct.append(accuracy * sum(e_batch))

                # 计算平均精度与损失
                if batch == epoch_size - 1:
                    accuracy = sum(correct) / float(sum(e_stop))
                    return (sum(costs) / float(epoch_size), accuracy)
            elif model.net_type == NetType.RNN_NV1:
                costs.append(cost)
                correct.append(accuracy)

                # 计算平均精度与损失
                if batch == epoch_size - 1:
                    cost_mean = sum(costs) / float(epoch_size)
                    accuracy_mean = sum(correct) / float(epoch_size)
                    return (cost_mean, accuracy_mean)

    print("训练完毕  " + str(epoch))

#队列版未完成
def run_batch_from_tfrecords(sess, coord, model, eval_op):
    if model.is_training and eval_op == model.train_op:
        count = 1
        iter = 0
        while not coord.should_stop():
            if count % conf.evaluate_freq != 0:
                _ = sess.run(model.train_op)

            else:
                coord.request_stop()
                print("第%d次测试精度" % (iter))
                evaluate_from_tfrecords(iter)
                coord.clear_stop()
                iter += 1
            count+=1

    else:
        accuracy_list = []
        cost_list = []
        try:
            while not coord.should_stop():
                cost, accuracy = sess.run([model.cost, model.accuracy])
                print(cost,accuracy)
                accuracy_list.append(accuracy)
                cost_list.append(cost)
        except tf.errors.OutOfRangeError:
            print("测试完成")
        acc_mean = sum(accuracy_list) / len(accuracy_list)
        cost_mean = sum(cost_list) / len(cost_list)
        return cost_mean, acc_mean

#nvn model
def rnn_nvn_model():
    #1 处理数据
    #2 设置模型
    #3 训练模型
    #4 测试模型

    global train_data
    global test_data
    global val_data
    #x shape = [序列长度，总的序列个数，特征长度]
    #y shape = [总的序列个数，1}
    #early_stop  shape = [总的序列个数]  [250,250,250,250,50,.........]
    #train_index  训练集的索引 [10,11,12,13,......]
    train_data, test_data, val_data, train_config, test_config, valid_config = loadData_rnn_nvn()

    minibatch = 0

    with tf.Session() as sess:
        tf.set_random_seed(0)

        #变量初始化
        initializer = tf.random_uniform_initializer(0,0.001)
        #正则化
        regularizer = layers.l2_regularizer(conf.l2_preparam)

        with tf.variable_scope("model",reuse=False,initializer=initializer,dtype=tf.float32): #,regularizer = regularizer):
            global train_model
            train_model = Model(conf,train_config)
        with tf.variable_scope("model",reuse=True,initializer=initializer,dtype=tf.float32): #,regularizer = regularizer):
            global test_model
            test_model = Model(conf,test_config)
        with tf.variable_scope("model",reuse=True,initializer=initializer,dtype=tf.float32): #,regularizer = regularizer):
            global valid_model
            valid_model = Model(conf,valid_config)

        saver  = None
        if conf.checkpoint or conf.restore:
            saver = tf.train.Saver()

        if conf.tensorboard:
            global writer
            writer = tf.summary.FileWriter(log_path, sess.graph)

        if not conf.restore:
            tf.global_variables_initializer().run()  # initialize all variables in the model
        else:
            saver.restore(sess, data_path + task)
            print("装载变量......")

        for i in range(conf.num_epochs):
            print("第 {0}次epoch".format(i))
            minibatch = run_batch(sess,train_model,train_data,train_model.train_op,minibatch)
            if (i+1)%10 == 0:
                saver.save(sess,data_path+task)

        if conf.checkpoint:
            save_path = saver.save(sess,data_path+task)

#nv1 model
def rnn_nv1_model(is_quick):
    global train_data
    global test_data
    global valid_data
    if is_quick:
        train_data,valid_data,test_data=load_data_rnn_nv1_quick(conf.num_classes)
    else:
        train_data, valid_data, test_data = load_data_rnn_nv1(conf.num_classes)
    #print("数据加载完毕......")
    train_conf = None
    valid_conf = None
    test_conf = None

    if conf.rnn_type == "lstm_b":
        train_conf = TrainingConfig(True,False,False,conf.batch_size,len_features,net_type=NetType.RNN_NV1,rnn_type=RNNType.LSTM_b)
        valid_conf = TrainingConfig(False,True,False,conf.batch_size,len_features,net_type=NetType.RNN_NV1,rnn_type=RNNType.LSTM_b)
        test_conf = TrainingConfig(False,False,True,conf.batch_size,len_features,net_type=NetType.RNN_NV1,rnn_type=RNNType.LSTM_b)
    elif conf.rnn_type == "gru_b":
        train_conf = TrainingConfig(True, False, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU_b)
        valid_conf = TrainingConfig(False, True, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU_b)
        test_conf = TrainingConfig(False, False, True, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                   rnn_type=RNNType.GRU_b)
    elif conf.rnn_type == "gru":
        train_conf = TrainingConfig(True, False, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU)
        valid_conf = TrainingConfig(False, True, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU)
        test_conf = TrainingConfig(False, False, True, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                   rnn_type=RNNType.GRU)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU90%的显存
    initializer = tf.random_uniform_initializer(0, conf.init_scale)

    with tf.variable_scope("model", reuse=False, initializer=initializer):
        global train_model
        train_model = Model(conf, train_conf)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        global test_model
        test_model = Model(conf, test_conf)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        global valid_model
        valid_model = Model(conf, valid_conf)

    minibatch = 0
    with tf.Session(config=config) as sess:

        saver = None
        if conf.checkpoint or conf.restore:
            saver = tf.train.Saver()

        if conf.tensorboard:
            global writer
            writer = tf.summary.FileWriter(log_path, sess.graph)

        if not conf.restore:
            tf.global_variables_initializer().run()  # initialize all variables in the model
        else:
            saver.restore(sess, data_path + task)
            print("装载变量......")

        LOGGER.training_log(str(conf.__dict__))
        LOGGER.training_log("activation = tanh")

        if is_quick:

            for i in range(conf.num_epochs):
                print("第 {0}次epoch".format(i))
                #minibatch = run_batch(sess, train_model, train_data, train_model.train_op, minibatch)
                run_batch_quick(sess,train_model,train_data,train_model.train_op,i)
                evaluate_model_quick(sess,i)
        else:
            for i in range(conf.num_epochs):
                print("第 {0}次epoch".format(i))
                #minibatch = run_batch(sess, train_model, train_data, train_model.train_op, minibatch)
                run_batch_all(sess,train_model,train_data,train_model.train_op,i)
                evaluate_model_all(sess,i)

#队列版  未完善
def rnn_nv1_model_tfrecord():
    global train_data
    global test_data
    global valid_data
    # train_data,valid_data,test_data=load_data_rnn_nv1_quick(conf.num_classes)
    # print("数据加载完毕......")
    train_conf = None
    valid_conf = None
    test_conf = None

    if conf.rnn_type == "lstm_b":
        train_conf = TrainingConfig(True, False, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.LSTM_b)
        valid_conf = TrainingConfig(False, True, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.LSTM_b)
        test_conf = TrainingConfig(False, False, True, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                   rnn_type=RNNType.LSTM_b)
    elif conf.rnn_type == "gru_b":
        train_conf = TrainingConfig(True, False, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU_b)
        valid_conf = TrainingConfig(False, True, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU_b)
        test_conf = TrainingConfig(False, False, True, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                   rnn_type=RNNType.GRU_b)
    elif conf.rnn_type == "gru":
        train_conf = TrainingConfig(True, False, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU)
        valid_conf = TrainingConfig(False, True, False, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                    rnn_type=RNNType.GRU)
        test_conf = TrainingConfig(False, False, True, conf.batch_size, len_features, net_type=NetType.RNN_NV1,
                                   rnn_type=RNNType.GRU)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU90%的显存
    initializer = tf.random_uniform_initializer(0, conf.init_scale)

    with tf.variable_scope("model", reuse=False, initializer=initializer):
        global train_model
        train_model = Model(conf, train_conf)

    # with tf.variable_scope("model", reuse=True, initializer=initializer):
    #     global test_model
    #     test_model = Model(conf, test_conf)
    #
    # with tf.variable_scope("model", reuse=True, initializer=initializer):
    #     global valid_model
    #     valid_model = Model(conf, valid_conf)


    train_filenames = np.array(util.search_file("interval_[1-5]_label_[0-3]_train.tfrecords", tfrecords_data_path))
    valid_filenames = np.array(util.search_file("interval_[1-5]_label_[0-3]_valid.tfrecords", tfrecords_data_path))
    test_filenames = np.array(util.search_file("interval_[1-5]_label_[0-3]_test.tfrecords", tfrecords_data_path))

    minibatch = 0
    with tf.Session(config=config) as sess:

        saver = None
        if conf.checkpoint or conf.restore:
            saver = tf.train.Saver()

        if conf.tensorboard:
            global writer
            writer = tf.summary.FileWriter(log_path, sess.graph)

        sess.run(tf.local_variables_initializer())
        if not conf.restore:
            sess.run(tf.global_variables_initializer())  # initialize all variables in the model
        else:
            saver.restore(sess, data_path + task)
            print("装载变量......")

        LOGGER.training_log(str(conf.__dict__))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord,start=True)
        #print("a")
        for i in range(10000):
            _  = sess.run(train_model.train_op)

        #run_batch_from_tfrecords(sess, coord, train_model, train_model.train_op)

        coord.request_stop()
        coord.join(threads)

def init_model_config(batch_size,len_features,net_type,rnn_type):
    train_conf = TrainingConfig(True, False, False, batch_size, len_features, net_type,rnn_type)
    valid_conf = TrainingConfig(False, True, False, batch_size, len_features, net_type,rnn_type)
    test_conf = TrainingConfig(False, False, True, batch_size, len_features, net_type,rnn_type)

    return train_conf,valid_conf,test_conf

#dataset版
def model_tfrecord_dataset(net_type,rnn_type):

    #初始文件路径等等
    #init_environment(net_type)

    train_conf,valid_conf,test_conf = init_model_config(conf.batch_size,len_features,net_type,rnn_type)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.25  # 占用GPU90%的显存
    initializer = tf.random_uniform_initializer(0, conf.init_scale)

    with tf.variable_scope("model", reuse=False, initializer=initializer):
        global train_model
        train_model = Model(conf, train_conf)

    with tf.variable_scope("model", reuse=True, initializer=initializer):
        global test_model
        test_model = Model(conf, test_conf)

    with tf.variable_scope("model", reuse=True, initializer=initializer):
        global valid_model
        valid_model = Model(conf, valid_conf)

    LOGGER.training_log(str(conf.__dict__))
    LOGGER.training_log(str(train_conf.activation))

    train_data_set = make_dataset_from_tfrecord_file(param.train_file_pattern,conf.batch_size,True,1)
    train_data_iterator = train_data_set.make_initializable_iterator()
    train_next_element = train_data_iterator.get_next()

    train_data_no_op_set = make_dataset_from_tfrecord_file(param.train_file_pattern, conf.batch_size, False, 1)
    train_data_no_op_iterator = train_data_no_op_set.make_initializable_iterator()
    train_no_op_next_element = train_data_no_op_iterator.get_next()


    valid_data_set = make_dataset_from_tfrecord_file(param.valid_file_pattern, conf.batch_size, False,1)
    valid_data_iterator = valid_data_set.make_initializable_iterator()
    valid_next_element = valid_data_iterator.get_next()

    test_data_set = make_dataset_from_tfrecord_file(param.test_file_pattern, conf.batch_size, False,1)
    test_data_iterator = test_data_set.make_initializable_iterator()
    test_next_element = test_data_iterator.get_next()

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        if conf.restore:
            saver.restore(sess,data_path + task + str(net_type.value)+str(rnn_type.value))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        for epoch in range(conf.num_epochs):
            sess.run(train_data_iterator.initializer)
            i = 0
            try:
                while True:
                    input, early, label = sess.run(fetches=train_next_element)
                    # 通过session每次从数据集中取值
                    #RNN_NV1网络
                    #print(input.shape)
                    if  net_type == NetType.RNN_NV1:

                        if input.shape[0] < 128:
                            print(input.shape)
                            break
                        input = np.transpose(input, [1, 0, 2])

                        sess.run(fetches=train_model.train_op, feed_dict={train_model.input_data: input,
                                                                          train_model.early_stop: early,
                                                                          train_model.targets: label})

                    #rnn_nvn网络
                    elif net_type == NetType.RNN_NVN:
                        if input.shape[0] < 128:
                            break
                        new_label = np.zeros([conf.batch_size,conf.exp_seq_len],np.int32)
                        for batch in range(conf.batch_size):
                            new_label[batch,0:early[batch]] = label[batch]
                            new_label[batch,early[batch]:] = 0
                        label = new_label
                        input = np.transpose(input, [1, 0, 2])
                        weight_sequence_loss = np.zeros([conf.batch_size,conf.exp_seq_len],np.float32)
                        for k in range(conf.batch_size):
                            weight_sequence_loss[k,0:early[k]] = 1

                        sess.run(fetches=train_model.train_op, feed_dict={  train_model.input_data:input,
                                                                            train_model.early_stop:early,
                                                                            train_model.targets:label,
                                                                            train_model.weight_sequence_loss:weight_sequence_loss})

                    #dnn 网络
                    elif net_type == NetType.DNN or net_type == NetType.DNN_MAXOUT:
                        list_input = []
                        list_label = []
                        for batch in range(input.shape[0]):
                            list_input.append(input[batch,0:early[batch],:])
                            new_label = np.zeros([early[batch]],np.int32)
                            new_label[:] = label[batch]
                            list_label.append(new_label)
                        input = np.concatenate(tuple(list_input),axis=0)
                        label = np.concatenate(tuple(list_label),axis=0)

                        sess.run(fetches=train_model.train_op, feed_dict={train_model.input_data:input,
                                                                          train_model.early_stop:early,
                                                                          train_model.targets:label})
                    print("训练集第%d个batch" %(i))
                    if i % 1256 == 0 and i >0:
                        train_cost = 0
                        train_acc = 0
                        valid_cost = 0
                        valid_acc = 0
                        #sess.run(train_data_no_op_iterator.initializer)
                        #train_cost, train_acc = evaluate_from_tfrecord_dataset(net_type, sess, train_model,train_no_op_next_element, tf.no_op(), i / 100)
                        #sess.run(valid_data_iterator.initializer)
                        #valid_cost,valid_acc = evaluate_from_tfrecord_dataset(net_type,sess,valid_model,valid_next_element,tf.no_op(),i/100)
                        sess.run(test_data_iterator.initializer)
                        test_cost,test_acc = evaluate_from_tfrecord_dataset(net_type,sess,test_model,test_next_element,tf.no_op(),i/100)
                        print("训练集cost:%f,acc:%f" %(train_cost,train_acc))
                        print("验证集cost:%f,acc:%f" %(valid_cost,valid_acc))
                        print("测试集cost:%f,acc:%f" %(test_cost,test_acc))
                        LOGGER.summary_log((train_cost,train_acc,valid_cost,valid_acc,test_cost,test_acc),i)
                    i = i + 1
            except tf.errors.OutOfRangeError:
                print("第%d  epoch end!" % (epoch))
            print("共 %d 个batch" %(i))
            print("第%d  epoch end!" % (epoch))
            save_path = saver.save(sess, data_path + task + str(net_type.value)+str(rnn_type.value))
            if save_path is not None:
                LOGGER.training_log(str(save_path))

#dataset 解析函数
def __parse_function(example_proto):
    feature = param.feature
    features = tf.parse_single_example(example_proto,feature)
    speed_sec = tf.reshape(tf.decode_raw(features[param.SPEED_SEC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    avg_speed = tf.reshape(tf.decode_raw(features[param.AVG_SPEED], tf.int64), [conf.exp_seq_len, param.WIDTH])
    std_speed = tf.reshape(tf.decode_raw(features[param.STD_SPEED], tf.int64), [conf.exp_seq_len, param.WIDTH])
    acc_sec = tf.reshape(tf.decode_raw(features[param.ACC_SEC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    mean_acc = tf.reshape(tf.decode_raw(features[param.MEAN_ACC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    std_acc = tf.reshape(tf.decode_raw(features[param.STD_ACC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    head = tf.reshape(tf.decode_raw(features[param.HEAD], tf.int64), [conf.exp_seq_len, param.WIDTH])
    head_mean = tf.reshape(tf.decode_raw(features[param.HEAD_MEAN], tf.int64), [conf.exp_seq_len, param.WIDTH])
    std_head = tf.reshape(tf.decode_raw(features[param.STD_HEAD], tf.int64), [conf.exp_seq_len, param.WIDTH])
    early = tf.cast(features[param.EARLY], tf.int32)
    label = tf.cast(features[param.LABEL], tf.int32)

    seq = tf.concat([speed_sec, avg_speed, std_speed, acc_sec, mean_acc, std_acc,head,head_mean,std_head], axis=1)
    seq_float32 = tf.cast(seq, tf.float32)

    return seq_float32,early,label

def __parse_function_3_features(example_proto):
    feature = param.feature
    features = tf.parse_single_example(example_proto, feature)
    speed_sec = tf.reshape(tf.decode_raw(features[param.SPEED_SEC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    avg_speed = tf.reshape(tf.decode_raw(features[param.AVG_SPEED], tf.int64), [conf.exp_seq_len, param.WIDTH])
    std_speed = tf.reshape(tf.decode_raw(features[param.STD_SPEED], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # acc_sec = tf.reshape(tf.decode_raw(features[param.ACC_SEC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # mean_acc = tf.reshape(tf.decode_raw(features[param.MEAN_ACC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # std_acc = tf.reshape(tf.decode_raw(features[param.STD_ACC], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # head = tf.reshape(tf.decode_raw(features[param.HEAD], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # head_mean = tf.reshape(tf.decode_raw(features[param.HEAD_MEAN], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # std_head = tf.reshape(tf.decode_raw(features[param.STD_HEAD], tf.int64), [conf.exp_seq_len, param.WIDTH])
    early = tf.cast(features[param.EARLY], tf.int32)
    label = tf.cast(features[param.LABEL], tf.int32)

    seq = tf.concat([speed_sec, avg_speed, std_speed], axis=1)
    seq_float32 = tf.cast(seq, tf.float32)

    return seq_float32, early, label

def __parse_function_6_features(example_proto):
    feature = param.feature
    features = tf.parse_single_example(example_proto, feature)
    speed_sec = tf.reshape(tf.decode_raw(features[param.SPEED_SEC], tf.int64), [conf.exp_seq_len, conf.discretization_width])
    avg_speed = tf.reshape(tf.decode_raw(features[param.AVG_SPEED], tf.int64), [conf.exp_seq_len, conf.discretization_width])
    std_speed = tf.reshape(tf.decode_raw(features[param.STD_SPEED], tf.int64), [conf.exp_seq_len, conf.discretization_width])
    acc_sec = tf.reshape(tf.decode_raw(features[param.ACC_SEC], tf.int64), [conf.exp_seq_len, conf.discretization_width])
    mean_acc = tf.reshape(tf.decode_raw(features[param.MEAN_ACC], tf.int64), [conf.exp_seq_len, conf.discretization_width])
    std_acc = tf.reshape(tf.decode_raw(features[param.STD_ACC], tf.int64), [conf.exp_seq_len, conf.discretization_width])
    # head = tf.reshape(tf.decode_raw(features[param.HEAD], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # head_mean = tf.reshape(tf.decode_raw(features[param.HEAD_MEAN], tf.int64), [conf.exp_seq_len, param.WIDTH])
    # std_head = tf.reshape(tf.decode_raw(features[param.STD_HEAD], tf.int64), [conf.exp_seq_len, param.WIDTH])
    early = tf.cast(features[param.EARLY], tf.int32)
    label = tf.cast(features[param.LABEL], tf.int32)

    seq = tf.concat([speed_sec, avg_speed, std_speed,acc_sec,mean_acc,std_acc], axis=1)
    seq_float32 = tf.cast(seq, tf.float32)

    return seq_float32, early, label

#创建dataset
def make_dataset_from_tfrecord_file(file_name_pattern,batch_size=32,is_shuffle=True,repeat = 1):
    filenames = util.search_file(file_name_pattern, tfrecords_data_path)
    filenames = np.array(filenames)
    perm = np.random.permutation(len(filenames))
    dataset = tf.data.TFRecordDataset(filenames[perm])
    if conf.num_features == 9:
        dataset = dataset.map(__parse_function)
    elif conf.num_features == 6:
        dataset = dataset.map(__parse_function_6_features)
    elif conf.num_features == 3:
        dataset = dataset.map(__parse_function_3_features)
    if is_shuffle:
        dataset = dataset.shuffle(100000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat)

    return dataset

#从npy获取数据 切割数据为预期长度的npy文件
def slice_seq(classes):
    # 分训练集与测试集 验证集 8：1：1
    train_data_all = None
    train_label_all = None
    train_early_all = None
    valid_data_all = None
    valid_label_all = None
    valid_early_all = None
    test_data_all = None
    test_label_all = None
    test_early_all = None
    features_arr_list = []
    index_arr_list = []
    label_arr_list = []
    data_file_name_exp = data_path +"transportation_mode"
    for i in range(classes):
        print("加载" + str(i))
        # data_file  = data_file_name +str(i) +".npy"
        index_df = pd.DataFrame(pd.read_csv(data_file_name_exp +"_"+ str(i) + "_seg_index.csv"))
        features_arr = np.load(data_file_name_exp + str(i) + ".npy")
        features_arr = features_arr[:, 0:len_features]
        index_arr = np.array(index_df.iloc[:, [1, 2]].T)
        # index shape = [2,总个数]
        # 第一维是第几段轨迹 第二维是在固定长度为exp_seq_len中的实际长度
        # data shape =[seq_nums,exp_seq_len,feature_len]   切出相等的数据长度 不足的padding
        (data, index_arr) = Data.slice_seq(features_arr, index_arr, conf.exp_seq_len)

        np.save(data_path+"slice_label" + str(i)+"_"+str(conf.exp_seq_len)+".npy",data)
        np.save(data_path+"slice_index"+str(i)+".npy",index_arr)

#分割数据集，合并数据 并写为npz文件
def partition_data_set(classes):
    out_data_path = "G:/all_data/"
    # 分训练集与测试集 验证集 8：1：1
    train_data_all = None
    train_label_all = None
    train_early_all = None
    valid_data_all = None
    valid_label_all = None
    valid_early_all = None
    test_data_all = None
    test_label_all = None
    test_early_all = None
    data_file_name_exp = data_path + "transportation_mode"
    for i in range(classes):
        data = np.load(data_path + "slice_label" + str(i) + "_" + str(conf.exp_seq_len) + ".npy")
        index_arr = np.load(data_path + "slice_index" + str(i) + ".npy")

        # 切割后删除features_arr index
        label_arr = np.zeros(shape=[index_arr.shape[1]], dtype=np.int32)
        label_arr[:] = i
        # features_arr_list.append(data)
        # index_arr_list.append(index)
        # label_arr_list.append(label)
        # 划分训练集，验证集，测试集
        print("划分训练集，验证集，测试集   " + str(i))
        seq_nums = index_arr.shape[1]
        # 控制变量
        np.random.seed(2)
        index_perm = np.random.permutation(range(seq_nums))
        train_count = int(np.floor(seq_nums * 0.8))
        valid_count = int(np.floor(seq_nums * 0.9))
        test_count = seq_nums
        train_index = index_perm[0:train_count]
        valid_index = index_perm[train_count + 1:valid_count]
        test_index = index_perm[valid_count + 1:seq_nums]

        # train_set valid_set test_set
        train_data = data[train_index, :, :]
        train_label = label_arr[train_index]
        train_early = index_arr[1, train_index]

        valid_data = data[valid_index, :, :]
        valid_label = label_arr[valid_index]
        valid_early = index_arr[1, valid_index]

        test_data = data[test_index, :, :]
        test_label = label_arr[test_index]
        test_early = index_arr[1, test_index]

        # 删除读取到的data.
        del data
        del label_arr
        del index_arr

        print("连接")
        if train_data_all is None:
            train_data_all = train_data
            train_label_all = train_label
            train_early_all = train_early

            valid_data_all = valid_data
            valid_label_all = valid_label
            valid_early_all = valid_early

            test_data_all = test_data
            test_label_all = test_label
            test_early_all = test_early
        else:
            train_data_all = np.concatenate((train_data_all, train_data), axis=0)
            train_label_all = np.concatenate((train_label_all, train_label), axis=0)
            train_early_all = np.concatenate((train_early_all, train_early), axis=0)

            valid_data_all = np.concatenate((valid_data_all, valid_data), axis=0)
            valid_label_all = np.concatenate((valid_label_all, valid_label), axis=0)
            valid_early_all = np.concatenate((valid_early_all, valid_early), axis=0)

            test_data_all = np.concatenate((test_data_all, test_data), axis=0)
            test_label_all = np.concatenate((test_label_all, test_label), axis=0)
            test_early_all = np.concatenate((test_early_all, test_early), axis=0)

    np.savez(out_data_path+"valid_data_set.npz",valid_data=valid_data_all,valid_label = valid_label_all,valid_early_stop = valid_early_all)
    del valid_label_all
    del valid_data_all
    del valid_early_all
    np.savez(out_data_path+"test_data_set.npz",test_data=test_data_all,test_label = test_label_all,test_early_stop = test_early_all)
    del test_label_all
    del test_early_all
    del test_data_all
    np.savez(out_data_path + "train_data_set.npz", train_data=train_data_all, train_label=train_label_all,train_early_stop=train_early_all)

def main():
        model_tfrecord_dataset(get_net_type(conf.net_type),get_rnn_type(conf.rnn_type))
    #rnn_nv1_model(False)

if __name__ == "__main__":
    #slice_seq(4)
    main()
    #partition_data_set(4)
