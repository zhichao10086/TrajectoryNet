import json
import numpy as np
import random
import math
import config
import pandas as pd
from param import FeatureName
from param import RNNType

b = np.array([[[1,2,3],[2,3,4]],[[7,8,9],[3,4,5]]])
c = np.reshape(b,[4,3])


df = pd.DataFrame(c,columns=["a","b","c"])
cc = df["a"]
nn = "a"
print(FeatureName.MEAN_ACC.value)
print(RNNType.GRU_b)
if "mean_acc" == FeatureName.MEAN_ACC.value:
    print(cc)
