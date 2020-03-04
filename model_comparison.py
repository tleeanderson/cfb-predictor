import glob
import argparse
import os.path as path
import team_game_statistics as tgs
import game
import model
from functools import reduce
import utilities as util
import os
import numpy as np

ALL_KEYS = reduce(lambda t1, t2: set(t1).union(set(t2)), model.FIELD_WIN_SEMANTICS.keys())

MODEL_COMPARISON = 'model_comparison'
AVERAGES = 'averages'
LABELS = 'labels'
DATA_CACHE_DIR = path.join(util.DATA_CACHE_DIR, MODEL_COMPARISON)
AVERAGES_CACHE = path.join(DATA_CACHE_DIR, AVERAGES)
LABELS_CACHE = path.join(DATA_CACHE_DIR, LABELS)

def parse_args():
    parser = argparse.ArgumentParser(description='Permute multiple feature sets')
    parser.add_argument('-i', '--input-directory', required=True, 
                        help='Top level directory of data')
    parser.add_argument('-ds', '--dir-suffix', required=True, help='')

    return parser.parse_args()

def select_columns(features, avgs):
    result = {}
    remove_keys = None
    for gid, ts in avgs.items():
        result[gid] = {}
        for tid, tv in ts.items():
            if remove_keys is None:
                remove_keys = set(tv.keys()).difference(set(features))
            result[gid][tid] = util.subset_of_map(full_map=tv, 
                                               take_out_keys=remove_keys)
    return result

def select_columns_season(features, sea_avgs):
    return {k: select_columns(features=features, avgs=sea_avgs[k]) for k in sea_avgs}

def feature_accuracy(feature_names, labels, avgs_sea):
    features = select_columns_season(features=feature_names, sea_avgs=avgs_sea)
    preds = model.predict_all_season(season_tgs=features)
    return model.accuracy_season(labels=labels, preds=preds, skip_keys={})

def fa_stats(fa):
    acc_vals = [sa['accuracy'] for sea, sa in fa.items()]
    acc_summary = {'average': np.average(acc_vals), 'min': np.min(acc_vals),
                   'max': np.max(acc_vals)}
    return {k: round(acc_summary[k] * 100, 2) for k in acc_summary}

def feature_accuracy_season(all_feats, labels, avgs_sea):
    return {f: fa_stats(fa=feature_accuracy(feature_names=f, 
                                            labels=labels, avgs_sea=avgs_sea)) for f in all_feats}

def print_fas(fas):
    for f, vm in sorted(iter(fas.items()), key=lambda x: x[1]['average'], reverse=True):
        print("{:50}{}".format(str(f), str(vm)))

if __name__ == '__main__':
    args = parse_args()
    all_dirs = glob.glob(path.join(args.input_directory, args.dir_suffix))
    tgs_data = tgs.tgs_data(dirs=all_dirs)
    game_data = game.game_data(dirs=all_dirs)
    avgs_sea = util.compute_or_cache(cache_dir=AVERAGES_CACHE, input_dir=args.input_directory, 
                                     comp_func=model.averages_season, comp_func_args={'tgs_data': tgs_data, 
                                                                                'game_data': game_data})
    points = select_columns_season(features=('Points',), sea_avgs=tgs_data)
    labels = util.compute_or_cache(cache_dir=LABELS_CACHE, input_dir=args.input_directory,
                                   comp_func=tgs.add_labels_season, comp_func_args={'team_game_stats': points})
    feature_names = tuple([(f,) for f in ALL_KEYS] \
                    + [('Kickoff Yard', 'Points', 'Red Zone Att', 'Rush Yard', 'Red Zone TD')] \
                    + [('Kickoff Yard', 'Points')])
    fas = feature_accuracy_season(all_feats=feature_names, labels=labels, avgs_sea=avgs_sea)
    print_fas(fas=fas)
