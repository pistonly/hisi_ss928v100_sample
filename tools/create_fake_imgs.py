import numpy as np

with open("fake_1920x1080_230.bin", "wb") as f:
    for i in range(100):
        img = np.zeros((1080, 1920), dtype=np.uint8)
        if i % 2 == 1:
            img += 230
        img.tofile(f)
    
