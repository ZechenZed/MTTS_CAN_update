import os
import pandas as pd


dir_path = "../Phase1_data/Videos/train-001_of_002"
res = []

for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
print(res)