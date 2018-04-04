from __future__ import division
import csv

import numpy as np
import math
import sklearn.preprocessing
import os
import time
import pandas as pd
import util


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

    @staticmethod
    def sovle_row_data():
        datadir = "G:/新建文件夹/Geolife Trajectories 1.3/Data/"

        valiable_user_data = open("./data/have_label_user.txt","r")
        user_list = valiable_user_data.readlines()
        for i in user_list:
            user_id = i[0:3]
            label_txt_name = datadir + user_id+"/labels.txt"
            label_file = open(label_txt_name,"r")
            #label文件 数据还是字符串
            list_label = label_file.readlines()[1:]
            #label_list 数据是label数组
            label_list = []
            for i in list_label:
                l = i[0:len(i)-1].split("\t")
                label_list.append(l)



            plt_path = datadir + user_id + "/Trajectory"
            list_plt_name = os.listdir(plt_path)

            user_data = datadir + user_id + "/userdata.csv"
            user_data_file = open(user_data,"w")

            label_time_index = 0

            #循环处理所有plt文件
            i = 0
            while(i < len(list_plt_name)):

                is_finish = False
                plt_name = list_plt_name[i]
                print("处理", plt_name)

                plt_file_name = plt_path + "/" + plt_name
                #plt_time_str = plt_name[0:4] + "/" +plt_name[4:6] + "/" +plt_name[6:8] +" " + plt_name[8:10] +":"+plt_name[10:12]+":"+plt_name[12:14]
                #plt_time = time.strptime(plt_time_str,'%Y/%m/%d %H:%M:%S')
                #if plt_time
                plt_file = open(plt_file_name,"r")
                data = plt_file.readlines()
                data = data[6:len(data)]

                #plt文件的起始时间
                plt_start_time_str = data[0]
                plt_end_time_str = data[-1]
                plt_start_time_list = plt_start_time_str[0:len(plt_start_time_str)-1].split(",")
                plt_start_time = time.strptime(plt_start_time_list[-2] + " " + plt_start_time_list[-1],'%Y-%m-%d %H:%M:%S')
                plt_end_time_list = plt_end_time_str[0:len(plt_end_time_str)-1].split(",")
                plt_end_time = time.strptime(plt_end_time_list[-2] + " " + plt_end_time_list[-1],'%Y-%m-%d %H:%M:%S')

                #label 当前起始时间
                label_start_time = time.strptime(label_list[label_time_index][0], '%Y/%m/%d %H:%M:%S')
                label_end_time = time.strptime(label_list[label_time_index][1], '%Y/%m/%d %H:%M:%S')

                #如果plt_end_time < 当前label_start_time 处理下一个plt文件
                if plt_end_time <= label_start_time:
                    i+=1
                    continue
                elif plt_start_time >= label_end_time :
                    #重复此次循环
                    i-=1
                    label_time_index += 1
                    if label_time_index > len(label_list)-1:
                        is_finish = True
                else:
                    #处理plt文件中的内容
                    print("处理有标签的文件",plt_name)

                    last_time = None
                    k = 0
                    while(k < len(data)):
                        line = data[k]
                        line_time_list = line[0:len(line)-1].split(",")
                        line_time = time.strptime(line_time_list[-2] + " " + line_time_list[-1],'%Y-%m-%d %H:%M:%S')
                        #print(line_time,label_start_time,label_end_time)

                        if line_time >= label_start_time and line_time <= label_end_time:
                            if k == 0:
                                last_time = line_time
                            else:
                                if line_time == last_time:
                                    last_time = line_time
                                    k+=1
                                    continue
                            result_line = user_id +"," + line[0:len(line)-1] + "," + label_list[label_time_index][-1] + "," +str(label_time_index)
                            user_data_file.write(result_line + "\n")
                            last_time = line_time
                            k+=1
                        elif line_time >label_end_time:

                            label_time_index += 1
                            if label_time_index > len(label_list)-1:
                                is_finish = True
                                break
                            label_start_time = time.strptime(label_list[label_time_index][0], '%Y/%m/%d %H:%M:%S')
                            label_end_time = time.strptime(label_list[label_time_index][1], '%Y/%m/%d %H:%M:%S')
                        elif line_time <label_start_time:
                            k+=1
                    #处理下一个文件
                #关闭当前plt文件
                if is_finish:
                    print("当前用户处理完毕",user_id)
                    plt_file.close()
                    break
                i+=1

            label_file.close()
            #plt_file.close()
            user_data_file.close()


    @staticmethod
    def caculate_feature():
        datadir = "G:/新建文件夹/Geolife Trajectories 1.3/Data/"
        feature_num = 10
        valiable_user_data = open("./data/have_label_user.txt", "r")
        user_list = valiable_user_data.readlines()
        for user in user_list:
            user_id = user[0:3]
            user_data_name = datadir + user_id + "/userdata.csv"
            print("开始处理",user_id)
            user_data_file = open(user_data_name,"r")

            # user_data_file = np.loadtxt(user_data_name,dtype=np.str,delimiter=",")
            # label_list = user_data_file[:,-1]
            # label_list = label_list.astype(int)
            # label_unique,label_index,label_count = np.unique(label_list, return_counts=True, return_index=True)
            # #print(label_unique,label_index,label_count)
            #
            #
            # for i in range(1):
            #     #一个label要使用的数组
            #     #result = np.empty(shape=[label_count[i],feature_num],dtype=np.str_)
            #     #一个label的索引在一个用户文件中
            #     start = label_index[i]
            #     end = label_index[i] + label_count[i]
            #     #一个label索引对应的原始数据
            #     data = user_data_file[start:end,:]
            #     #经纬度 以及时间
            #     lat_lon_time = data[:,[1,2,5]]
            #     #将user_id,经纬度赋值给结果数组
            #     #result[:,0:3] = data[:,0:3]
            #
            #     #计算特征  速度 加速度  开始点没有速度，第一个点没有加速度， 所以最后数组比原始数组少两个点
            #     for i in range(1,len(lat_lon_time)):
            #         dis = util.jwd2dis(lat_lon_time[i][0],lat_lon_time[i][1],lat_lon_time[i-1][0],lat_lon_time[i-1][1])
            #         t = util.timestamp2second(lat_lon_time[i],lat_lon_time[i-1])
            #
            #     print(lat_lon_time)



            # #user_data = user_data_file.readlines()
            #列名
            col_name = ["user_id","lat","lon","non-use","alt","timestamp","date","time","label","label_count"]
            #原始数据
            raw_data_df = pd.DataFrame(pd.read_csv(user_data_file,header=None,names=col_name))
            #结果列名
            result_col_name = ["user_id","lat","lon","speed_sec","acc_sec","std_speed","avg_speed","mean_acc","date","time","label","seg_label"]
            #结果数据
            result_df = pd.DataFrame(columns=result_col_name)

            #通过标签分组轨迹
            label_gp = raw_data_df.groupby(by=col_name[-1])

            for label_count,group in label_gp:
                #print(group)
                #print(len(group.index))
                #temp_result = pd.DataFrame(columns = result_col_name)
                #特征数组
                print("label_count",label_count)
                if (group.index[-1] - group.index[0]) < 2:
                    print("丢弃本组数据")
                    continue
                feature_arr = np.zeros(shape=[group.index[-1] - group.index[0] +1,5],dtype=np.float64)
                #print(group)
                #print(len(group.index))
                offset =  group.index[0]
                for ii in  group.index[1:]:
                    #row_result = pd.Series(index=result_col_name)
                    dis = util.jwd2dis(group.loc[ii,"lat"],group.loc[ii,"lon"],group.loc[ii-1,"lat"],group.loc[ii-1,"lon"])
                    t = util.timestamp2second(group.loc[ii,"timestamp"],group.loc[ii-1,"timestamp"])

                    feature_arr[ii - offset][0] = dis/t
                    if(ii > offset+1):
                        #a = (v1-v0)/t
                        feature_arr[ii- offset][1] = (feature_arr[ii- offset][0] - feature_arr[ii-1-offset][0]) / t



                avg_speed = np.mean(feature_arr[2:,0],axis=0)
                acc_mean = np.mean(feature_arr[2:,1],axis=0)
                std_speed = np.std(feature_arr[2:,0],axis=0)
                feature_arr[2:,2] = std_speed
                feature_arr[2:,3] = avg_speed
                feature_arr[2:,4] = acc_mean
                feature_arr = feature_arr[2:,:]

                #print(feature_arr)
                result = pd.DataFrame(columns=result_col_name)
                #result["user_id"] = group["user_id"][2:len(group.index)]
                start = group.index[0] + 2
                end = group.index[-1]
                result["user_id"] = group.loc[start:end,"user_id"]
                result["lat"] = group.loc[start:end,"lat"]
                result["lon"] = group.loc[start:end,"lon"]
                #print(result.info(),length,feature_arr.shape)
                result["speed_sec"] = feature_arr[:,0]
                result["acc_sec"] = feature_arr[:,1]
                result["std_speed"] = feature_arr[:,2]
                result["avg_speed"] = feature_arr[:,3]
                result["mean_acc"] = feature_arr[:,4]
                result["date"] = group.loc[start:end,"date"]
                result["time"] = group.loc[start:end,"time"]
                result["label"] = util.switch_mode(group.loc[start,"label"])
                result["seg_label"] = user_id +" " + str(group.loc[start,"label_count"])
                #一组label最终结果dataframe
                result_df = result_df.append(result)



            result_df.index = range(0,result_df.shape[0])

            result_df.to_csv(datadir + user_id +"/user_features.csv",index=False)
            user_data_file.close()








if __name__ == "__main__":
    Data.sovle_row_data()
    Data.caculate_feature()