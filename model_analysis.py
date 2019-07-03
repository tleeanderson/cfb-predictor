import sys
import glob
import pickle
import numpy as np
import os.path as path
import pprint

def read_data_file(**kwargs):
    file_name = kwargs['file_name']
    with open(file_name, 'rb') as fh:
        eval_data = pickle.load(fh)

    return eval_data

def all_data(**kwargs):
    model_dir = kwargs['model_dir']

    data = {}
    for eval_file in glob.glob(path.join(model_dir, '*.model_eval')):
        data[eval_file] = read_data_file(file_name=eval_file)
    
    return data

def eval_data(**kwargs):
    data = kwargs['data']

    eval_data = {}
    for ef, stats in data.iteritems():
        for season, sea_stat in stats.iteritems():
            if season not in eval_data:
                eval_data[season] = {}
            for key in set(sea_stat.keys()).difference({'Split', 'global_step'}):
                if key not in eval_data[season]:
                    eval_data[season][key] = []
                eval_data[season][key].append(sea_stat[key])
    
    return eval_data

def metadata(**kwargs):
    d = kwargs['eval_data']

    meta = {}
    for season, data in d.iteritems():
        if season not in meta:
            meta[season] = {}
        for key, val in data.iteritems():
            meta[season][key + '_avg'] = np.average(val)
            meta[season][key + '_var'] = np.var(val)
            meta[season][key + '_max'] = mx = np.max(val)
            meta[season][key + '_min'] = mi = np.min(val)
            meta[season][key + '_range'] = mx - mi
            meta[season]['num_points'] = len(val)

    return meta

def main(argv):
    if len(argv) == 2:
        data = all_data(model_dir=argv[1])
        evaluation = eval_data(data=data)

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(metadata(eval_data=evaluation))
    else:
        print("Usage: %s [eval_data_dir]" % ('./' + str(argv[0])))
        

if __name__ == '__main__':
    main(sys.argv)
