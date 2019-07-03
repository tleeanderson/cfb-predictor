import numpy as np
import utilities as util
import team_game_statistics as tgs
import os.path as path
import game
import operator as op
import model
import glob
import sys

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        print("v " + str(v))
        # for attr, av in v.iteritems():
        #     print("attr: %s av: %s" % (str(attr), str(av)))
        #print("len(v) " + str(len(v)))
        raw_input() 

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
    directory, prefix, no_pred = kwargs['directory'], kwargs['prefix'], 'no_pred'
    
    model_acc = {}
    for data_dir in glob.glob(path.join(directory, prefix)):
        stats = team_game_stats(directory=data_dir)
        game_data = game_stats(directory=data_dir)

        preds = model.predict_all(team_game_stats=stats, game_infos=game_data, no_pred_key=no_pred)
        stats_labels = tgs.add_labels(team_game_stats=stats)

        accuracy = model.accuracy(tg_stats=stats_labels, predictions=preds, corr_key='correct', 
                                  incorr_key='incorrect', total_key='total', skip_keys={no_pred}, 
                                  acc_key='accuracy')
        model_acc[data_dir] = accuracy
    
    return model_acc

def print_summary(**kwargs):
    model_acc = kwargs['model_acc']

    for data_dir, ma in model_acc.iteritems():
        print("directory: %s, model_accuracy: %s" % (str(data_dir), str(ma)))
        
    accs = map(lambda x: x[1]['accuracy'], model_acc.iteritems())
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

def main(args):
    if len(args) == 3:
        #read in data
        input_directory, prefix = args[1], args[2]
        model_acc = evaluate_model(directory=input_directory, prefix=args[2])
        print_summary(model_acc=model_acc)
        #loop_through(data=tgs.add_labels(team_game_stats=stats))
    else:
        print("usage: ./%s [top_level_dir] [data_dir_prefix]" % (sys.argv[0]))

if __name__ == '__main__':
    main(sys.argv)
