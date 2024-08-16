import numpy as np

for i in range(100):
    try:
        txt1 = f"/home/liuyang/Documents/haisi/my_samples/tools/packet_{i}.txt"
        txt2 = f"/home/liuyang/Documents/haisi/my_samples/tools/tmp/packet_{i}.txt"

        d1 = np.loadtxt(txt1)
        d2 = np.loadtxt(txt2)
        print(i, abs(d1[2:-1] - d2[3:]).sum())
        print(i, abs(d1 - d2).sum())
    except:
        continue
