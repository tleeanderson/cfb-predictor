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

STATIC_ANALYSIS = 'static_analysis'
DATA_CACHE_DIR = path.join(util.DATA_CACHE_DIR, STATIC_ANALYSIS)
ACC_BY_DATE_CACHE = path.join(DATA_CACHE_DIR, 'histogram')
ACCURACY_CACHE = path.join(DATA_CACHE_DIR, 'accuracy')
FIGURE_DIR = path.join(util.FIGURE_DIR, STATIC_ANALYSIS)
YEAR_FROM_DIR = lambda s: path.basename(s).replace('cfbstats-com-', '')[:4]

def evaluate_model(dirs, model_fn):
    """Evaluates a given model_fn over a given set of dirs.

    Args:
         dirs: season directories
         model_fn: function to evaluate data
    
    Returns: map of season dir name name to model_fn 
             applied to each season
    """

    no_pred = 'no_pred'
    result = {}
    for data_dir in dirs:
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

def model_meta_data(acc_data):
    """Given model accuracy data, computes max, min, range, and average.

    Args:
         acc_data: map of model data
    
    Returns: map of model metadata
    """

    accs = map(lambda x: x[1]['accuracy'], acc_data.iteritems())
    model_meta = {}
    model_meta['max_accuracy'] = mx = max(accs)
    model_meta['min_accuracy'] = mi = min(accs)
    model_meta['range_acc'] = mx - mi
    model_meta['average'] = np.average(accs)

    return model_meta

def accuracy_bar_chart_by_season(acc_data, model_meta_data, fig_loc):
    """Creates accuracy bar chart by season. All games are included.

    Args:
         acc_data: map of model data by season
         model_meta_data: metadata concerning acc_data
    
    Returns: None
    """
    ak, mia, ma = 'accuracy', 'min_accuracy', 'max_accuracy'
    avg, t = 'average', 'total'

    th = model_meta_data[avg] * 100
    acc_ys = np.array(map(lambda s: acc_data[s][ak] * 100, acc_data))
    ng_ys = np.array(map(lambda s: acc_data[s][t], acc_data))
    x = sorted(map(YEAR_FROM_DIR, acc_data.keys()))
    avg_label = avg + '=' + str(round(model_meta_data[avg] * 100, 2))

    fig, ax = plt.subplots()
    fig.set_figwidth(14)
    ax.bar(x, acc_ys, 0.8, color='b')  
    plt.axhline(y=th, linewidth=1.5, color='k', **{'label': avg_label})
    plt.ylabel('Percentage Accuracy', fontsize=16)
    plt.xlabel('Season', fontsize=16)
    plt.title('Accuracy by Season', fontsize=20)
    plt.legend([avg_label], loc=0)
    for acc, g in zip(enumerate(acc_ys), enumerate(ng_ys)):
        ai, av = acc
        ax.text(ai - 0.2, 1, str(round(av, 2)) + '%', color='white', fontweight='bold')
        gi, gv = g
        ax.text(gi - 0.2, 5, str(gv) + 'g', color='white', fontweight='bold')

    plt.savefig(path.join(fig_loc, 'accuracy_by_season.png'))
        
def accuracy_bar_chart_by_date_per_season(acc_by_date, fig_loc):
    """Creates bar chart of model by each day in season.

    Args:
         acc_by_date: map of model data by date per season
    
    Returns: None
    """
    avg, num_games_avg = 'acc_avg', 'num_games_avg'

    for s, s_acc in acc_by_date.iteritems():
        day_accs = map(lambda d: (d, round(s_acc[d]['accuracy'] * 100, 2), s_acc[d]['total']), s_acc)
        day_accs.sort(key=lambda x: du.parse(x[0]))
        xs = map(lambda t: t[0], day_accs)
        acc_ys = map(lambda t: t[1], day_accs)
        ng_ys = map(lambda t: t[2], day_accs)
        acc_avg = round(np.average(acc_ys), 2)
        ng_avg = round(np.average(ng_ys), 2)
        acc_avg_label = avg + '=' + str(acc_avg)
        ng_avg_label = num_games_avg + '=' + str(ng_avg)

        fig, ax = plt.subplots()
        fig.set_figwidth(18)
        ax.bar(xs, acc_ys, 0.8, color='b')
        ax.bar(xs, ng_ys, 0.8, color='g')

        plt.axhline(y=acc_avg, linewidth=2.0, color='black', **{'label': acc_avg_label})
        plt.axhline(y=ng_avg, linewidth=2.0, color='purple', **{'label': ng_avg_label})

        plt.ylabel('Percentage Accuracy', fontsize=16)
        plt.xlabel('Day', fontsize=16)
        plt.title('Accuracy by Day Within Season', fontsize=20)
        plt.legend([acc_avg_label, ng_avg_label], loc=0)
        for a, g in zip(enumerate(acc_ys), enumerate(ng_ys)):
            ai, av = a
            ax.text(ai - 0.3, 1, str(round(av, 2)) + '%', color='white', fontweight='bold')
            gi, gv = g
            ax.text(gi - 0.3, 5, str(gv) + 'g', color='white', fontweight='bold')

        plt.savefig(path.join(fig_loc, str(YEAR_FROM_DIR(s)) + '_accuracy_by_date.png'))

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
    if not path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    if args.accuracy_by_date:
        print('Histograming accuracy by date')
        acc_by_date, cs = util.model_data(comp_func_args={'model_fn': model.accuracy_by_date}, cache_dir=ACC_BY_DATE_CACHE, 
                                      **md_args)
        util.print_cache_reads(coll=cs, data_origin=ACC_BY_DATE_CACHE)
        filt_abd = {}
        for s, s_acc in acc_by_date.iteritems():
            filt_abd[s] = model.filter_by_total(acc_by_date=s_acc, lowest_val=10)

        accuracy_bar_chart_by_date_per_season(acc_by_date=filt_abd, fig_loc=FIGURE_DIR)
    else:
        print('Calculating accuracy by season')
        acc, cs = util.model_data(comp_func_args={'model_fn': model.accuracy}, cache_dir=ACCURACY_CACHE, **md_args)
        util.print_cache_reads(coll=cs, data_origin=ACCURACY_CACHE)
        accuracy_bar_chart_by_season(acc_data=acc, model_meta_data=model_meta_data(acc_data=acc), fig_loc=FIGURE_DIR)
    print("Figures written to %s" % (path.abspath(FIGURE_DIR)))

if __name__ == '__main__':
    main(sys.argv)
