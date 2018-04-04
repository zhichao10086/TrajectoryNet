
import time

class Log(object):

    def __init__(self,path):
        self.train_log_path = path
        second = time.localtime(time.time())
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S',second)
        #summary_log_file 初始化   由当前时间命名
        self.summary_log_file_name =  "summary"+time_str +".csv"
        self.summary_log_file = open(path +self.summary_log_file_name,"w+")

        #training_log_file 由training+当前时间命名
        self.train_log_file_name =  "training" + time_str + ".csv"
        self.train_log_file = open(path +self.train_log_file_name,"w+")

        self.addheader()

    def addheader(self):
        self.summary_log_file.write("iteration, trainLoss, valLoss, testLoss, trainAcc, valAcc, testAcc\n")
        #self.train_log_file.write("iteration, trainLoss,trainAcc\n")

    def summary_log(self,data,batch_iter):
        (cost_train, acc_train, cost_test, acc_test, cost_val, acc_val) = data
        self.summary_log_file.write("{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}, {5:0.3f}, {6:0.3f}\n".format(batch_iter, cost_train, cost_val, cost_test, acc_train, acc_val, acc_test))
        self.summary_log_file.flush()


    def training_log(self,data):
        #trainLoss,trainAcc =
        self.train_log_file.write(data)
        self.train_log_file.write("\n")
        self.train_log_file.flush()


    def close(self):
        self.summary_log_file.close()
        self.train_log_file.close()


if __name__ == "__main__":

    print(str(3)+"dd")