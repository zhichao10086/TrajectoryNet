import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.rcParams['font.family']='SimHei'

logdir = "./logdir/shiyanxiuzheng/"

def read_csv_file(name):

    file = open(name)

    df = pd.read_csv(file)

    result_arr = np.array(df,np.float64)

    return result_arr


def draw_chart(arr):
    plt.plot(arr[:,-1])
    plt.show()

def draw_4_features():
    temp_dir = "12features/"
    features_3 = read_csv_file(logdir +temp_dir+"RNN_NV1_GRU_b_3.csv")
    features_6 = read_csv_file(logdir + temp_dir+"RNN_NV1_GRU_b_6.csv")
    features_9 = read_csv_file(logdir + temp_dir+"RNN_NV1_GRU_b_9.csv")
    features_12 = read_csv_file(logdir + temp_dir+"RNN_NV1_GRU_b_12.csv")

    limit = 20
    x = range(1,limit+1)

    plt.plot(x,features_3[0:limit, -1], "bx-", label="3个特征")
    plt.plot(x,features_6[0:limit, -1], "rx-", label="6个特征")
    plt.plot(x,features_9[0:limit, -1], "gx-", label="9个特征")
    plt.plot(x,features_12[0:limit, -1], "yx-", label="12个特征")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("RNN_Nv1 双层双向MaxoutGRU模型不同特征的精度（测试集）")
    plt.savefig(logdir +temp_dir +"rnn_nv1_features")


    plt.show()

def draw_width():
    temp_dir = "width/"
    features_3 = read_csv_file(logdir +temp_dir+"RNN_NV1_GRU_b_10.csv")
    features_6 = read_csv_file(logdir + temp_dir+"RNN_NV1_GRU_b_20.csv")
    features_9 = read_csv_file(logdir + temp_dir+"RNN_NV1_GRU_b_30.csv")
    features_12 = read_csv_file(logdir + temp_dir+"RNN_NV1_GRU_b_40.csv")

    limit = 20
    x = range(1,limit+1)

    plt.plot(x,features_3[0:limit, -1], "bx-", label="10")
    plt.plot(x,features_6[0:limit, -1], "rx-", label="20")
    plt.plot(x,features_9[0:limit, -1], "gx-", label="30")
    plt.plot(x,features_12[0:limit, -1], "yx-", label="40")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.96)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("RNN_Nv1 双层双向MaxoutGRU模型 离散宽度（测试集）")
    plt.savefig(logdir +temp_dir +"rnn_nv1_width")


    plt.show()

def draw_dnn():
    dnn = read_csv_file(logdir + "result_dnn/dnn.csv")
    dnn_dropout = read_csv_file(logdir + "result_dnn/dnn_dropout.csv")
    dnn_maxout = read_csv_file(logdir + "result_dnn/dnn_maxout.csv")

    x = range(1,20)

    plt.plot(x,dnn[:, -1], "bx-", label="dnn")
    plt.plot(x,dnn_dropout[:, -1], "rx-", label="dnn_dropout")
    plt.plot(x,dnn_maxout[:, -1], "gx-", label="dnn_maxout")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.5, 1)
    plt.xlim(0, 21)
    plt.legend(loc=1)
    plt.title("三种DNN模型的精度（测试集）")
    plt.savefig(logdir + "result_dnn/dnn_3")

    plt.show()

def rnn_3():

    temp_dir = "3_rnn/"

    lstm = read_csv_file(logdir + temp_dir+"RNN_NV13 2.csv")
    maxoutgru = read_csv_file(logdir + temp_dir+ "RNN_NV13 4.csv")
    normal_gru = read_csv_file(logdir + temp_dir+ "RNN_NV13 6.csv")

    limit = 29
    x = range(1,limit+1)

    plt.plot(x,lstm[0:limit, -1], "bx-", label="lstm")
    plt.plot(x,normal_gru[0:limit, -1], "rx-", label="normal_gru")
    plt.plot(x,maxoutgru[0:limit, -1], "gx-", label="maxout_gru")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=1)
    plt.title("三种RNN模型的精度（测试集）")
    plt.savefig(logdir +temp_dir +"3_rnn")

    plt.show()

def gru_2():
    temp_dir = "3_rnn/"

    gru = read_csv_file(logdir + temp_dir + "RNN_NV13 3.csv")
    gru_b= read_csv_file(logdir + temp_dir + "RNN_NV13 4.csv")

    limit = 48
    x = range(1, limit + 1)

    plt.plot(x, gru[0:limit, -1], "bx-", label="单向MaxoutGRU")
    plt.plot(x, gru_b[0:limit, -1], "rx-", label="双向MaxoutGRU")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit + 2)
    plt.legend(loc=1)
    plt.title("单双向MaxoutGRU模型的精度（测试集）")
    plt.savefig(logdir + temp_dir + "gru_2")

    plt.show()

def rnn_nvn():
    temp_dir = "3nvn_gru/"

    gru = read_csv_file(logdir + temp_dir+"RNN_NVN_GRU.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "RNN_NVN_GRU_b 3 feature.csv")
    gru_b_9 = read_csv_file(logdir + temp_dir+ "RNN_NVN_GRU_b 9 feature.csv")

    limit = 30
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="单向MaxoutGRU")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="双向MaxoutGRU 3个特征")
    plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="双向MaxoutGRU 9个特征")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=1)
    plt.title("RNN_NVN模型的精度（测试集）")
    plt.savefig(logdir +temp_dir +"3_nvn_gru")

    plt.show()

def draw_3_model():
    temp_dir = "3_model/"

    gru = read_csv_file(logdir + temp_dir+"dnn no dropout.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "rnn_nvn.csv")
    gru_b_9 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")

    limit = 39
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="双层DNN")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="双层双向MaxoutGRU RNN_NVN")
    plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="双层双向MaxoutGRU RNN_NV1")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("三种模型的精度（测试集）")
    plt.savefig(logdir +temp_dir +"3_model")

    plt.show()

def draw_nvn_2():
    temp_dir = "3nvn_gru/"

    gru = read_csv_file(logdir + temp_dir+"RNN_NVN_LSTM_b.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "RNN_NVN_GRU_b 9 feature.csv")
    #gru_b_9 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")

    limit = 39
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="双层双向LSTM RNN_NVN")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="双层双向MaxoutGRU RNN_NVN")
    #plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="双层双向MaxoutGRU RNN_NV1")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("RNN_NvN模型两种网络结构的精度（测试集）")
    plt.savefig(logdir +temp_dir +"2_rnn")

    plt.show()

def draw_baifenwei():
    temp_dir = "baifenweiduibi/"

    gru = read_csv_file(logdir + temp_dir+"baifenwei95.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "baifenwei99.csv")
    #gru_b_9 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")

    limit = 30
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="百分位95")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="百分位99")
    #plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="双层双向MaxoutGRU RNN_NV1")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.84, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("百分位对比（测试集）")
    plt.savefig(logdir +temp_dir +"baifenwei")

    plt.show()

def draw_batch_size():

    temp_dir = "3_model/"

    gru = read_csv_file(logdir + temp_dir+"rnn_nv1.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "rnn_nv1_256.csv")
    #gru_b_9 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")

    limit = 30
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="batch size 128")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="batch size 256")
    #plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="双层双向MaxoutGRU RNN_NV1")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.84, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("batch size 对比（测试集）")
    plt.savefig(logdir +temp_dir +"batch size")

    plt.show()

def draw_hidden():
    temp_dir = "3_model/"

    gru = read_csv_file(logdir + temp_dir+"rnn_nv1_hidden_50.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "rnn_nv1_hidden_200.csv")
    gru_b_9 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")

    limit = 41
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="hidden size 50")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="hidden size 200")
    plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="hidden size 100")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("HIDDEN SIZE（测试集）")
    plt.savefig(logdir +temp_dir +"2_rnn")

    plt.show()

def draw_activation():
    temp_dir = "3_model/"

    gru = read_csv_file(logdir + temp_dir+"rnn_nv1_tahn.csv")
    gru_b_3 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")
    #gru_b_9 = read_csv_file(logdir + temp_dir+ "rnn_nv1.csv")

    limit = 66
    x = range(1,limit+1)

    plt.plot(x,gru[0:limit, -1], "bx-", label="tahn")
    plt.plot(x,gru_b_3[0:limit, -1], "rx-", label="sigmod")
    #plt.plot(x,gru_b_9[0:limit, -1], "gx-", label="hidden size 100")

    plt.xlabel("mini-batch")
    plt.ylabel("accuarcy")
    plt.ylim(0.8, 0.95)
    plt.xlim(0, limit+2)
    plt.legend(loc=0)
    plt.title("MaxoutGRU的两种activation（测试集）")
    plt.savefig(logdir +temp_dir +"activation")

    plt.show()

if __name__ == "__main__":
    draw_width()