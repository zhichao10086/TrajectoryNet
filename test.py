import json
import numpy as np
import random

def get_match_index(mmsi, target):
    unique_mmsi = np.unique(mmsi)
    result = np.concatenate([np.where(mmsi == unique_mmsi[i]) for i in target], axis=1)[0]
    return result


mmsi = np.array([10,10,10,20,20,20,20,40,40,40,40,50,50,50,50,50,60,60,60,60,70,70])
test_mmsi = np.array([0,1,2,3])
bb = [25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25]
unique_mmsi = np.unique(mmsi)

result = [np.where(mmsi == unique_mmsi[i]) for i in test_mmsi]
nn = np.concatenate(result, axis=1)[0]

v = mmsi[[0,1,2,3,4,6]]
b = np.maximum([1,2,3],[0,3,4])
b = np.random.randint(0,10,[2,3])
c = np.random.randint(0,10,[2,3])

a = [b,c]
b = mmsi[nn]
np.random.shuffle(b)
vvv = np.random.permutation(b)
b = np.array([[[1,2,3],[2,3,4]],[[7,8,9],[3,4,5]]])
c = np.reshape(b,[4,3])
test_vessel = random.sample(range(23),6)
t = None
while (1):
    t = np.random.randint(0, 23, 1)
    if t not in test_vessel:
        break
l = range(1,4)
list = [ [None] *5]*6
v= np.random.randint(0,10,size=[6,5])
v = v.astype(dtype=np.str)
b = np.array(["123","rr"])

print(type(b[0]))