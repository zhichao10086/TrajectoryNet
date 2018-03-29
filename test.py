import json
import numpy as np

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
print(c)
print(np.reshape(c,[2,2,3]))