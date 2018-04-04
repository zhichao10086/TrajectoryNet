from math import radians, cos, sin, asin, sqrt

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