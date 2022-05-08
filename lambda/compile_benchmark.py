import argparse
import os
from glob import glob
import pandas as pd


list_gpus = {
    'a6000': ('RTX A6000', 300, 5785),
    'rtx8000': ('RTX 8000', 260, 6900),
    '3080': ('3080', 320, 1642),
}

list_benchmarks = {
    'YOLOv5n': (640),
    'YOLOv5s': (640),
    'YOLOv5m': (640),
    'YOLOv5l': (640),
    'YOLOv5x': (640),
    'YOLOv5n6': (1280),
    'YOLOv5s6': (1280),
    'YOLOv5m6': (1280),
    'YOLOv5l6': (1280),
    'YOLOv5x6TTA': (1280),  
}


list_methods = {
    'PyTorch': 'PyTorch',
    'TorchScript': 'TorchScript',
    'ONNX': 'ONNX',
    'OpenVINO': 'OpenVINO',
    'TensorRT': 'TensorRT',
    'CoreML': 'CoreML',
    'TF': 'TensorFlow SavedModel',
    'TFGraphDef': 'TensorFlow GraphDef',
    'TFLite': 'TensorFlow Lite',
    'TFEdgeTPU': 'TensorFlow Edge TPU',
    'TFJS': 'TensorFlow.js',
}


def main():
    parser = argparse.ArgumentParser(description='Gather benchmark results.')
    parser.add_argument('--path', type=str, default='benchmarks',
                        help='path to the benchmarks folder')    
    parser.add_argument('--method', type=str, default='PyTorch',
                        choices=['PyTorch', 'TorchScript', 'ONNX', 'OpenVINO', 'TensorRT', 'CoreML', 'TF', 'TFGraphDef', 'TFLite', 'TFEdgeTPU', 'TFJS'],
                        help='Choose inference method')
    args = parser.parse_args()
    
    last_N = 13
    
    list_folders = glob(args.path + "/*/", recursive = True)
    
    list_configs = [list_gpus[key][0] for key in list_gpus]
        
    columns = []
    columns.append('watt')
    columns.append('price')

    for benchmark_name, benchmark_imgsz in sorted(list_benchmarks.items()):
        columns.append(benchmark_name)

    df = pd.DataFrame(index=list_configs, columns=columns)
    df = df.fillna(-1.0)
    df.index.name = 'name_gpu'

    
    for folder in list_folders:
        gpu = folder.split('/')[-2].split('-')[0]
        df.at[list_gpus[gpu][0], 'watt'] = list_gpus[gpu][1]
        df.at[list_gpus[gpu][0], 'price'] = list_gpus[gpu][2]
    
        for benchmark in list_benchmarks:
            with open(os.path.join(folder, benchmark + '.txt')) as file:
                for line in (file.readlines() [-last_N:]):
                    items = ' '.join(line.strip().split())
                    if list_methods[args.method] in items:
                        t = items.split(' ')[-1]
                        throughput = 1000/float(t)
                        df.at[list_gpus[gpu][0], benchmark] = format(throughput, '.2f')
                        break

    df.to_csv(os.path.join(args.path, args.method + '.csv'))
    
if __name__ == "__main__":
    main()