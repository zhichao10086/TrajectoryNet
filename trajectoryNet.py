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
import param

conf = config.Config("data/config.json")
log_path = "./logdir/transportation_feature_en_1_interval_1&2_expand_all/"
data_path = "./data/transportation_feature_en_1_interval_1&2_expand_all/"
task = conf.task
LOGGER = Log(log_path)
len_features = conf.num_features * 20

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

#直接读取整个data  noTranspose
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

def rnn_nv1_model(is_quick):
    global train_data
    global test_data
    global valid_data
    train_data,valid_data,test_data=load_data_rnn_nv1_quick(conf.num_classes)
    print("数据加载完毕......")
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

    minibatch = 0
    with tf.Session(config=config) as sess:
        initializer = tf.random_uniform_initializer(0, conf.init_scale)

        with tf.variable_scope("model",reuse=False,initializer=initializer):
            global train_model
            train_model = Model(conf,train_conf)
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            global test_model
            test_model = Model(conf,test_conf)
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            global valid_model
            valid_model = Model(conf,valid_conf)

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

            #if (i + 1) % 10 == 0:
            #    saver.save(sess, data_path + task)

def main():
    rnn_nv1_model(True)

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

if __name__ == "__main__":
    #slice_seq(4)
    main()
    #partition_data_set(4)