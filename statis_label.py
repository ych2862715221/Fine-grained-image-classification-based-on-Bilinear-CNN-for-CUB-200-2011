import sys
import numpy as np
label = np.zeros(200)
print(label.shape)

datas = open(sys.argv[1]).readlines()
for data in datas:
    data = int(data.strip().split(' ')[-1])
    label[data] += 1

print(label)
