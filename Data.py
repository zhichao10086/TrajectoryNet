from __future__ import division
import csv

import numpy as np
import math
import sklearn.preprocessing


class Data:
    @staticmethod
    def splitDataset(mmsi, tr_mmsi, vl_tmmsi):
        test_index = Data.get_match_index(mmsi, tr_mmsi)
        val_index = Data.get_match_index(mmsi, vl_tmmsi)
        train_index = np.delete(np.array(range(len(mmsi))), np.concatenate([test_index, val_index]))
        return (train_index, test_index, val_index)

    @staticmethod
    def randomSplitDataset(mmsi, train_perc=0.5, val_perc=0.1):
        mmsi = np.array(mmsi)
        seq_len = mmsi.shape[0]
        test_perc = 1 - train_perc - val_perc
        rdn_perm = np.random.permutation(seq_len)
        train_index = rdn_perm[0:int(seq_len * train_perc)]
        test_index = rdn_perm[int(seq_len * train_perc): int(seq_len * (train_perc + test_perc))]
        val_index = rdn_perm[int(seq_len * (train_perc + test_perc)): seq_len]
        return (train_index, test_index, val_index)

    @staticmethod
    def get_match_index(mmsi, target):
        unique_mmsi = np.unique(mmsi)
        result = np.concatenate([np.where(mmsi == unique_mmsi[i]) for i in target], axis=1)[0]
        return result

    @staticmethod
    def upsample(data, cls, times):
        (X_train, y_train, stop_train) = data
        labels = [set(i) for i in y_train]
        samples = [cls in i for i in labels]
        sample_index = np.where(samples)[0]
        sample_x = np.repeat(X_train[:, sample_index, :], times - 1, axis=1)
        sample_y = np.repeat(y_train[sample_index, :], times - 1, axis=0)
        sample_stop = np.repeat(stop_train[sample_index], times - 1, axis=0)
        X_train = np.concatenate((X_train, sample_x), axis=1)
        y_train = np.vstack((y_train, sample_y))
        stop_train = np.hstack((stop_train, sample_stop))
        return (X_train, y_train, stop_train)

    # cut sequence into smaller sequences specified by the conf
    # 将序列切成指定长度的
    @staticmethod
    def reorganizeSeq(x, y, mmsi, exp_seq_len):
        num_features = x.shape[2]
        # 总共可以切出的序列个数
        num_total_seq = int(sum([math.ceil(i) for i in mmsi[1] / exp_seq_len]))
        new_data = np.zeros((num_total_seq, exp_seq_len, num_features))
        new_label = np.zeros((num_total_seq, exp_seq_len))
        # 0行存放编号 1行存放序列长度
        new_mmsi = np.zeros((2, num_total_seq)).astype(int)
        count = 0
        for v in range(len(mmsi[0])):  # iterate each vessel
            # 每个用户的数据
            # print v
            vessel_data = x[v]
            vessel_lab = y[v]
            # 用户编号
            vessel_mmsi = mmsi[0][v]
            # print(mmsi[0][v])
            # get full sequences first
            # 各个用户能切出的序列个数
            num_full_seq = mmsi[1][v] // exp_seq_len
            if num_full_seq:
                # full_seq的shape为当前用户的（总个数，序列长度，特征）
                full_seq = vessel_data[0:num_full_seq * exp_seq_len].reshape((num_full_seq, exp_seq_len, num_features))
                full_lab = vessel_lab[0:num_full_seq * exp_seq_len].reshape((num_full_seq, exp_seq_len))
                new_data[count:(count + num_full_seq)] = full_seq
                new_label[count:(count + num_full_seq)] = full_lab
                new_mmsi[0][count:(count + num_full_seq)] = vessel_mmsi
                new_mmsi[1][count:(count + num_full_seq)] = exp_seq_len
                count += num_full_seq

            # 序列切片多出来的长度保存起来
            remain_seq = np.zeros((exp_seq_len, num_features))
            remain_seq[0:(mmsi[1][v] - num_full_seq * exp_seq_len)] = vessel_data[num_full_seq * exp_seq_len:mmsi[1][v]]
            remain_lab = np.zeros(exp_seq_len)
            remain_lab[0:(mmsi[1][v] - num_full_seq * exp_seq_len)] = vessel_lab[num_full_seq * exp_seq_len:mmsi[1][v]]
            new_data[count] = remain_seq
            new_label[count] = remain_lab
            new_mmsi[0][count] = vessel_mmsi
            new_mmsi[1][count] = mmsi[1][v] - num_full_seq * exp_seq_len
            count += 1
        return (new_data, new_label, new_mmsi)