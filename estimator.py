import main as temp_lib
import sys
import model
import team_game_statistics as tgs
import numpy as np
import copy

def check(**kwargs):
    tga = kwargs['tga']

    keys = {}
    for gid, stats in tga.iteritems():
        for tid, stats in stats.iteritems():
            keys = stats.keys()
            break

    for gid, stats in tga.iteritems():
        for tid, stats in stats.iteritems():
            if set(keys) != set(stats.keys()):
                print("mismatch found: gid %s, tids %s len keys: %d, len stats: %d" 
                      % (gid, tid, len(keys), len(stats.keys())))

        # keys = check(tga=avgs)
        # more_keys = set(keys)
        # keys = map(lambda k: k + "_0", keys)
        # more_keys = map(lambda k: k + "_1", more_keys)
        # print(set(cols.keys()).difference(set(keys).union(set(more_keys))))
    
    return keys

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        print("v " + str(v))
        # for tid, stat in v.iteritems():
        #     print("tid " + str(tid))
        #     print("stat len " + str(len(stat)))
        #print("len(v) " + str(len(v)))
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

def main(args):
    if len(args) == 2:
        gs = temp_lib.game_stats(directory=args[1])
        team_stats = temp_lib.team_game_stats(directory=args[1])

        avgs = averages(team_game_stats=team_stats, game_infos=gs, skip_fields=model.UNDECIDED_FIELDS)
        team_stats = {k: team_stats[k] for k in avgs.keys()}
        
        labels = tgs.add_labels(team_game_stats=team_stats)
        data = input_data(game_averages=avgs, labels=labels)

        #loop_through(data=data[0])
        #print("len features: %d, len labels: %d" % (len(data[0]), len(data[1])))
    else:
        print("usage: ./%s [top_level_dir] [data_dir_prefix]" % (sys.argv[0]))

if __name__ == '__main__':
    main(sys.argv)
