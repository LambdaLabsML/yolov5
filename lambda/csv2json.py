import os

import pandas
import glob
import json

DATA_PATH = 'benchmarks'
DATA_FORMAT = '.csv'
SKIPROWS = None

pandas.set_option("display.max_colwidth", 1000)
for f in glob.glob(os.path.join(DATA_PATH, '*' + DATA_FORMAT)):
    method_name = os.path.splitext(os.path.basename(f))[0]
    dir_name = os.path.dirname(f)
    f_out = dir_name + '/yolov5-inference-latency-' + method_name + '.json'
    df = pandas.read_csv(f,
                         index_col=False,
                         skiprows=SKIPROWS,
                         header=0)
    df.to_json(f_out, orient='records')