import logging
import pandas as pd
import numpy as np
import json
data = pd.read_csv('Data/kuaishou_data_es.csv')
embed_data = data['audio_text']
def ddict(d):
    if not isinstance(d, dict) and not isinstance(d, list):
        print(d)
        return
    if isinstance(d, dict):
        for k in d:
            print(k)
            ddict(d[k])
    else:
        for k in d:
            ddict(k)

for i in embed_data[0:1]:
    ddict(json.loads(i))
# df_test = pd.DataFrame(columns=['a'])
# test_arr = np.array([i for i in range(10000)])
# df_test['a'] = [test_arr.tolist()]
# df_test.to_csv('test.csv')
# df_2 = pd.read_csv('test.csv')
# print((df_2['a'][0]))
# print(json.loads(df_2['a'][0]))