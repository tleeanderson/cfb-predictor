import main as temp_lib
import sys
import model
import team_game_statistics as tgs
import numpy as np
import copy
from dateutil import parser as du
import math
import random
from random import randint

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        #print("v " + str(v))
        # for tid, stat in v.iteritems():
        #     print("tid " + str(tid))
        #     print("stat len " + str(len(stat)))
        print("len(v) " + str(len(v)))
        #print("keys of v " + str(v.keys()))
        #raw_input()

def averages(**kwargs):
    team_game_stats, game_infos, skip_fields = kwargs['team_game_stats'], kwargs['game_infos'],\
                                               kwargs['skip_fields']

    game_avgs = {}
    for gid in team_game_stats.keys():
        avgs = model.team_avgs(game_code_id=gid, game_data=game_infos, tg_stats=team_game_stats)
        if len(avgs) == 2:
            game_avgs[gid] = {}
            for tid, stats in avgs.iteritems():
                game_avgs[gid][tid] = {k: stats[k] for k in set(stats.keys()).difference(skip_fields)}
        else:
            pass
        
    return game_avgs

def input_data(**kwargs):
    game_avgs, input_labels = kwargs['game_averages'], kwargs['labels']

    features = {}
    labels = []
    for gid, team_avgs in game_avgs.iteritems():
        for ta, feature_team_id, fid in zip(team_avgs.iteritems(), ['_0', '_1'], [0, 1]):
            tid, stats = ta
            for name, value in stats.iteritems():
                stat_key = name + feature_team_id
                if stat_key not in features:
                    features[stat_key] = []
                features[stat_key].append(value)
            if input_labels[gid]['Winner']['Team Code'] == tid:
                labels.append(fid)

    for k in features.keys():
        features[k] = np.array(features[k])

    return features, np.array(labels)

def histogram_games(**kwargs):
    game_infos, game_stats, histo_key = kwargs['game_infos'], kwargs['game_stats'], kwargs['histo_key']

    histo = {}
    for gid in game_stats.keys():
        info = game_infos[gid]
        if info[histo_key] not in histo:
            histo[info[histo_key]] = []
        histo[info[histo_key]].append(gid)
    
    return histo

def sort_by(**kwargs):
    input_map, key_sort = kwargs['input_map'], kwargs['key_sort']

    lis = list(input_map.iteritems())
    lis.sort(key=key_sort)

    return lis

def split_data(**kwargs):
    gh, sp, shc = kwargs['game_histo'], float(kwargs['split_percentage']), kwargs['histo_count']

    train = []
    test = []
    train_divi = True
    for k, count in shc.iteritems():
        if count == 1:
            if train_divi:
                train.append(gh[k][0])
                train_divi = False
            else:
                test.append(gh[k][0])
                train_divi = True
        elif count == 2:
            num = randint(0, 1)
            train.append(gh[k][num])
            test.append(gh[k][int(not num)])
        else:
            train_split = int(round(count * sp))
            test_split = count - train_split
            if test_split == 0:
                train_split = int(math.ceil(float(count) / 2))
            ind_range = set(range(len(gh[k])))
            train_ind = set(random.sample(ind_range, train_split))
            test_ind = ind_range.difference(train_ind)
            train += [gh[k][e] for e in train_ind]
            test += [gh[k][e] for e in test_ind]

    return train, test

def split_by_date(**kwargs):
    split, gs = kwargs['split'], kwargs['game_info']

    tdh = {}
    for gid in split:
        if gs[gid]['Date'] not in tdh:
            tdh[gs[gid]['Date']] = []
        tdh[gs[gid]['Date']].append(gid)

    tdh = {k: len(tdh[k]) for k in tdh.keys()}
    return sort_by(input_map=tdh, key_sort=lambda x: du.parse(x[0]))

def visualize_split(**kwargs):
    split, gs, tg = kwargs['split'], kwargs['game_info'], kwargs['total_games']

    train = split_by_date(split=split[0], game_info=gs)
    test = split_by_date(split=split[1], game_info=gs)  

    for tr, tst in zip(train, test):
        print("train: %s\ttest: %s" % (str(tr), str(tst)))

    print("Defensive test. Intersection of train and test should be empty, (intersection train test): " 
          + str(set(split[0]).intersection(set(split[1]))))
    print("Addition of splits should equal total games. Total games: %s Addition: %s" 
          % (str(tg), str(len(split[0]) + len(split[1]))))

def main(args):
    if len(args) == 2:
        gs = temp_lib.game_stats(directory=args[1])
        team_stats = temp_lib.team_game_stats(directory=args[1])

        avgs = averages(team_game_stats=team_stats, game_infos=gs, skip_fields=model.UNDECIDED_FIELDS)
        team_stats = {k: team_stats[k] for k in avgs.keys()}        
        labels = tgs.add_labels(team_game_stats=team_stats)        
        histo = histogram_games(game_infos=gs, game_stats=avgs, histo_key='Date')            
        split = split_data(game_histo=histo, split_percentage=0.85, 
                           histo_count={k: len(histo[k]) for k in histo.keys()})

        #data = input_data(game_averages=avgs, labels=labels)
        
        #print("len features: %d, len labels: %d" % (len(data[0]), len(data[1])))
    else:
        print("usage: ./%s [top_level_dir] [data_dir_prefix]" % (sys.argv[0]))

if __name__ == '__main__':
    main(sys.argv)
