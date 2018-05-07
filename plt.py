import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_csv_file(name):

    file = open(name)

    df = pd.read_csv(file)

    result_arr = np.array(df,np.float64)

    return result_arr


def draw_chart(arr):
    plt.plot(arr[:,-1])
    plt.show()




if __name__ == "__main__":
    arr = read_csv_file("./logdir/shiyan/dnn/summary2018-05-05-15-38-59.csv")
    draw_chart(arr)