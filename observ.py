import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dir = "C://TrafficMonitor//TrafficMonitor//Result//LONG_GANG//speed_160//baseline//explore_2500_n_steps_500_lr_5e-05_batch_size_350//logs//1.monitor.csv"
# df = pd.read_csv(dir, skiprows = 1)
# 
# x = df['r'].values
# 
# plt.plot(x)
# plt.show()


import os

for root, dirs, files in os.walk("C:\Users\hzysg\Desktop\RL\TrafficMonitor-wrapper_modification\Result\LONG_GANG\speed_160\fusion\explore_700_n_steps_512_lr_0.0005_batch_size_32\tensorboard"):
    for file in files:
        if file.startswith('events.out.tfevents'):
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            print(f"{size} bytes - {path}")