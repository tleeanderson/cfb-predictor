import numpy as np
import utilities as util
import team_game_statistics as tgs
import os.path as path
import game
import model
import glob
import sys
from dateutil import parser as du
import matplotlib.pyplot as plt
import argparse
import time
import os

DATA_CACHE_DIR = path.join(util.DATA_CACHE_DIR, 'static_analysis')
ACC_BY_DATE_CACHE = path.join(DATA_CACHE_DIR, 'histogram')
ACCURACY_CACHE = path.join(DATA_CACHE_DIR, 'accuracy')

def evaluate_model(**kwargs):
    ds, no_pred, model_fn = kwargs['dirs'], 'no_pred', kwargs['model_fn']
    
    result = {}
    for data_dir in ds:
        stats = tgs.team_game_stats(directory=data_dir)
        game_data = game.game_stats(directory=data_dir)
        preds = model.predict_all(team_game_stats=stats, game_infos=game_data, no_pred_key=no_pred)
        stats_labels = tgs.add_labels(team_game_stats=stats)
        result[data_dir] = model_fn(tg_stats=stats_labels,
                                          predictions=preds,
                                          game_info=game_data, 
                                          correct_key='correct', 
                                          incorrect_key='incorrect', 
                                          total_key='total', 
                                          acc_key='accuracy', 
                                          skip_keys={no_pred})
    return result

def model_meta_data(**kwargs):
    acc = kwargs['acc_data']

    accs = map(lambda x: x[1]['accuracy'], acc.iteritems())
    model_meta = {}
    model_meta['max_accuracy'] = mx = max(accs)
    model_meta['min_accuracy'] = mi = min(accs)
    model_meta['range_acc'] = mx - mi
    model_meta['average'] = np.average(accs)

    return model_meta
    
def season_accuracy_graph(**kwargs):
    acc, mmd, ak, mia, ma, avg = kwargs['acc_data'], kwargs['model_meta_data'], 'accuracy', 'min_accuracy', 'max_accuracy',\
                                 'average'

    th = mmd[avg] * 100
    values = np.array(map(lambda s: acc[s][ak] * 100, acc))
    x = sorted(map(lambda s: path.basename(s).replace('cfbstats-com-', '')[:4], acc.keys()))
    avg_label = avg + '=' + str(round(mmd[avg] * 100, 2))

    fig, ax = plt.subplots()
    fig.set_figwidth(14)
    ax.bar(x, values, 0.8, color='b')  
    plt.axhline(y=th, linewidth=1.5, color='k', **{'label': avg_label})
    plt.ylabel('Percentage Accuracy', fontsize=16)
    plt.xlabel('Season', fontsize=16)
    plt.title('Accuracy by season', fontsize=20)
    plt.legend([avg_label], loc=0)
    for i, v in enumerate(values):
        ax.text(i - 0.15, v / 2, str(round(v, 2)), color='white', fontweight='bold')

    plt.show()
        
def histogram_by_date(**kwargs):
    abd = kwargs['acc_by_date']

    for s, s_acc in abd.iteritems():
        plt.figure(num=s)
        day_accs = map(lambda d: (d, s_acc[d]['accuracy']), s_acc)
        day_accs.sort(key=lambda x: du.parse(x[0]))
        ticks = np.arange(len(s_acc))
        plt.bar(ticks, map(lambda d: d[1], day_accs), align='center', alpha=0.5)
        plt.xticks(ticks, map(lambda d: d[0], day_accs))
        plt.ylabel('Accuracy')
        plt.title(s + ' Accuracy by Date')
    plt.show()

def main(args):
    parser = argparse.ArgumentParser(description='Run static analysis on the cfb dataset')
    parser.add_argument('--input-directory', required=True, help='Top level directory of data')
    parser.add_argument('--dir-suffix', required=True, help='')
    parser.add_argument('--accuracy-by-date', required=False, action='store_true', 
                        help='Histograms the predictions of the model by date')
    args = parser.parse_args()

    all_dirs = glob.glob(path.join(args.input_directory, args.dir_suffix))
    md_args = {'cache_read_func': util.read_from_cache, 'cache_write_func': util.write_to_cache, 
               'all_dirs': all_dirs, 'comp_func': evaluate_model, 'context_dir': args.input_directory}
    if args.accuracy_by_date:
        print('Histograming accuracy by date')
        acc_by_date, cs = util.model_data(comp_func_args={'model_fn': model.accuracy_by_date}, cache_dir=ACC_BY_DATE_CACHE, 
                                      **md_args)
        util.print_cache_reads(coll=cs, data_origin=ACC_BY_DATE_CACHE)
        filt_abd = {}
        for s, s_acc in acc_by_date.iteritems():
            filt_abd[s] = model.filter_by_total(acc_by_date=s_acc, lowest_val=10)

        histogram_by_date(acc_by_date=filt_abd)
    else:
        print('Calculating accuracy by season')
        acc, cs = util.model_data(comp_func_args={'model_fn': model.accuracy}, cache_dir=ACCURACY_CACHE, **md_args)
        util.print_cache_reads(coll=cs, data_origin=ACCURACY_CACHE)
        season_accuracy_graph(acc_data=acc, model_meta_data=model_meta_data(acc_data=acc))

if __name__ == '__main__':
    main(sys.argv)
