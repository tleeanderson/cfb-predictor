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

def team_game_stats(**kwargs):
    input_directory = kwargs['directory']

    tgs_file = path.join(input_directory, tgs.FILE_NAME)
    team_game_stats = util.read_file(input_file=tgs_file, func=tgs.csv_to_map)
    converted_tgs = tgs.alter_types(type_mapper=tgs.type_mapper, 
                                    game_map=team_game_stats)
    
    return converted_tgs

def game_stats(**kwargs):
    input_directory = kwargs['directory']

    game_file = path.join(input_directory, game.FILE_NAME)
    game_data = util.read_file(input_file=game_file, func=game.csv_to_map)
    
    return game_data

def evaluate_model(**kwargs):
    directory, dir_suffix, no_pred, model_fn = kwargs['directory'], kwargs['dir_suffix'], 'no_pred',\
                                          kwargs['model_fn']
    
    result = {}
    for data_dir in glob.glob(path.join(directory, dir_suffix)):
        stats = team_game_stats(directory=data_dir)
        game_data = game_stats(directory=data_dir)
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

def print_summary(**kwargs):
    acc = kwargs['acc']

    for data_dir, ma in acc.iteritems():
        print("directory: %s, model_accuracy: %s" % (str(data_dir), str(ma)))
        
    accs = map(lambda x: x[1]['accuracy'], acc.iteritems())
    model_meta = {}
    model_meta['max_accuracy'] = max(accs)
    model_meta['min_accuracy'] = min(accs)
    model_meta['range_acc'] = model_meta['max_accuracy'] - model_meta['min_accuracy']
    model_meta['average'] = np.average(accs)

    print("\n")
    out_name = 'MODEL STATISTICS'
    print(('*' * 30) + out_name + ('*' * 30))
    for stat, val in model_meta.iteritems():
        print("%s: %s" % (str(stat), str(val)))
    print(('*' * 30) + ('*' * len(out_name)) + ('*' * 30))

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

    if args.accuracy_by_date:
        print("Histograming accuracy by date")
        acc_by_date = evaluate_model(directory=args.input_directory, dir_suffix=args.dir_suffix, 
                                           model_fn=model.accuracy_by_date)
        filt_abd = {}
        for s, s_acc in acc_by_date.iteritems():
            filt_abd[s] = model.filter_by_total(acc_by_date=s_acc, hi_total=10)

        histogram_by_date(acc_by_date=filt_abd)
    else:
        print("Calculating accuracy by season")
        acc = evaluate_model(directory=args.input_directory, dir_suffix=args.dir_suffix, model_fn=model.accuracy)
        print_summary(acc=acc)

if __name__ == '__main__':
    main(sys.argv)
