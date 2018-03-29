import unittest
import datetime
import time

class Log(object):

    def __init__(self,path):
        self.train_log_path = path


        self.summary_log_file = open(path + "111.csv","w+")
        self.addheader()

    def addheader(self):
        self.log_file.write("iteration, trainLoss, valLoss, testLoss, trainAcc, valAcc, testAcc\n")

    def summary_log(self,data,batch_iter):
        (cost_train, acc_train, cost_test, acc_test, cost_val, acc_val) = data
        self.log_file.write("{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}, {5:0.3f}, {6:0.3f}\n".format(batch_iter, cost_train, cost_val, cost_test, acc_train, acc_val, acc_test))
        self.log_file.flush()


    def training_log(self,data,batch_iter):
        fdf =3
        pass


    def close(self):
        self.log_file.close()



if __name__ == "__main__":
    print(time.time())
    b = time.localtime(time.time())
    print( time.strftime('%Y-%m-%d %H:%M:%S',b))