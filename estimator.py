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
import tensorflow as tf

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
        for ta, feature_team_id, fid in zip(team_avgs.iteritems(), ['-0', '-1'], [0, 1]):
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

def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    
    logits = tf.layers.dense(net, units=params['num_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def train_input_fn(features, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((dict(features), labels))\
                             .shuffle(1000).repeat().batch(batch_size)
    
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs = features if labels is None else (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset

def run_model(**kwargs):
    avgs, split, labels = kwargs['team_avgs'], kwargs['split'], kwargs['labels']

    train_features, train_labels = input_data(game_averages={gid: avgs[gid] for gid in split[0]}, 
                                              labels=labels)
    test_features, test_labels = input_data(game_averages={gid: avgs[gid] for gid in split[1]}, 
                                            labels=labels)
    feature_columns = []
    for key in train_features.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(model_fn=model_fn, params={'feature_columns': feature_columns, 
                                                                   'hidden_units': [84, 16], 
                                                                   'num_classes': 2})
    classifier.train(input_fn=lambda: train_input_fn(train_features, train_labels, BATCH_SIZE),
                     steps=TRAIN_STEPS)
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_features, 
                                                                     test_labels, BATCH_SIZE))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

BATCH_SIZE = 20
TRAIN_STEPS = 10000

def rename_keys(**kwargs):
    team_stats = copy.deepcopy(kwargs['team_stats'])

    for gid, teams in team_stats.iteritems():
        for tid, stats in teams.iteritems():
            for name in stats.keys():
                stats[name.replace(' ', '-')] = stats.pop(name)
    return team_stats

def main(args):
    if len(args) == 2:
        gs = temp_lib.game_stats(directory=args[1])
        team_stats = temp_lib.team_game_stats(directory=args[1])
        team_stats = rename_keys(team_stats=team_stats)

        avgs = averages(team_game_stats=team_stats, game_infos=gs, skip_fields=model.UNDECIDED_FIELDS)
        team_stats = {k: team_stats[k] for k in avgs.keys()}        
        labels = tgs.add_labels(team_game_stats=team_stats)        
        histo = histogram_games(game_infos=gs, game_stats=avgs, histo_key='Date')            
        split = split_data(game_histo=histo, split_percentage=0.85, 
                           histo_count={k: len(histo[k]) for k in histo.keys()})
        run_model(team_avgs=avgs, split=split, labels=labels)
    else:
        print("usage: ./%s [top_level_dir] [data_dir_prefix]" % (sys.argv[0]))

if __name__ == '__main__':
    main(sys.argv)
