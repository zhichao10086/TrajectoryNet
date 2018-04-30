from math import radians, cos, sin, asin, sqrt
import  os
from glob import glob

def jwd2dis(lat1,lon1,lat2,lon2):
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def timestamp2second(time1,time2):

    return abs(time1-time2)*3600*24

def switch_mode(str):
    str = str.strip()
    if(str == "bike"):
        return "0"

    if(str == "car"):
        return "1"

    if (str == "walk"):
        return "2"

    if (str == "bus"):
        return "3"

    if (str == "train"):
        return "4"

    if (str == "subway"):
        return "5"

    if (str == "airplane"):
        return "6"

    if (str == "taxi"):
        return "7"
    if (str == "boat"):
        return "8"
    if (str == "run"):
        return "9"
    if (str == "motorcycle"):
        return "10"
    else:
        print(str)
        return "11"

def rename_file():

    data_dir = "G:/新建文件夹/Geolife Trajectories 1.3/Data/"

    valiable_user_data = open("./data/have_label_user.txt", "r")
    user_list = valiable_user_data.readlines()
    for i in user_list[1:]:
        user_id = i[0:3]
        data_txt_name = data_dir + user_id + "/userdata.csv"
        features_name = data_dir+user_id + "/user_features.csv"
        new_data_name = data_dir + user_id + "/userdata_interval_1.csv"
        new_features_name = data_dir + user_id + "/user_features_interval_1.csv"
        os.rename(data_txt_name,new_data_name)
        os.rename(features_name,new_features_name)

def delete_file():
    data_dir = "G:/新建文件夹/Geolife Trajectories 1.3/Data/"

    valiable_user_data = open("./data/have_label_user.txt", "r")
    user_list = valiable_user_data.readlines()
    for i in user_list:
        user_id = i[0:3]
        data_txt_name = data_dir + user_id + "/user_features_interval_2.csv"

        os.remove(data_txt_name)

def search_file(pattern,path):
    paths  = glob(os.path.join(path,pattern))
    filenames = [ path.split("\\")[1]  for path in paths]
    filenames = [os.path.join(path,name) for name in filenames]
    return filenames
if __name__ == "__main__":
    print(search_file("interval_[0-1]_*_train.tfrecords","G:/all_data/tfrecords/"))