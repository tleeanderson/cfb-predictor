import model
import team_game_statistics as tgs
import numpy as np
from dateutil import parser as du
import math
import random
from random import randint
import tensorflow as tf
import glob
import os
import os.path as path
import distribution_analysis as da
from proto import estimator_pb2
from google.protobuf import text_format
import argparse
from google.protobuf.json_format import MessageToDict
import uuid
import pickle
import game
import utilities as util

TF_FEATURE_NAME = lambda f: f.replace(' ', '-')
BATCH_SIZE = 20
TRAIN_STEPS = 3000
DATA_CACHE_DIR = path.join(util.DATA_CACHE_DIR, 'estimator')

def averages(**kwargs):
    """Computes averages for both teams in all games in team_game_stats.

    Args:
         team_game_stats: map of game ids to statistics
         game_infos: home, away etc. information for a game, see game.csv
         skip_fields: fields to take out when storing statistics in final result
    
    Returns: tuple of game avgs and game ids that could not be averaged
    """
    team_game_stats, game_infos, skip_fields = kwargs['team_game_stats'], kwargs['game_infos'],\
                                               kwargs['skip_fields']

    game_avgs = {}
    no_avgs = []
    for gid in team_game_stats.keys():
        avgs = model.team_avgs(game_code_id=gid, game_data=game_infos, tg_stats=team_game_stats)
        if len(avgs) == 2:
            game_avgs[gid] = {}
            for tid, stats in avgs.iteritems():
                game_avgs[gid][tid] = {k: stats[k] for k in set(stats.keys()).difference(skip_fields)}
        else:
            no_avgs.append(gid)
        
    return game_avgs, no_avgs

def binary_classification_data(**kwargs):
    """(Deprecated) Creates input for binary classification model. Each 
       example is organized such that each column represents the input
       and the corresponding index in labels is the output.
       

    Args:
         game_averages: map of game ids to the averages for both teams
         labels: map containing labels for each game
    
    Returns: tuple of features and labels
    """
    game_avgs, input_labels = kwargs['game_averages'], kwargs['labels']

    features = {}
    labels = []
    for gid, team_avgs in game_avgs.iteritems():
        for ta, feature_team_id, fid in zip(team_avgs.iteritems(), ['-0', '-1'], [0, 1]):
            tid, stats = ta
            for name, value in stats.iteritems():
                stat_key = TF_FEATURE_NAME(name + feature_team_id)
                if stat_key not in features:
                    features[stat_key] = []
                features[stat_key].append(value)
            if input_labels[gid]['Winner']['Team Code'] == tid:
                labels.append(fid)

    for k in features.keys():
        features[k] = np.array(features[k])

    return features, np.array(labels)

def regression_data(**kwargs):
    """Creates input for regression model. Each example is organized
       such that each column represents the input and the corresponding
       index in labels is the output. Each label has shape [2] and each
       value is a score for each team.

    Args:
         game_averages: map of game ids to the averages for both teams
         labels: map containing labels for each game
    
    Returns: tuple of features and labels
    """
    game_avgs, input_labels, ps, w = kwargs['game_averages'], kwargs['labels'], 'Points', 'Winner'

    features = {}
    labels = []
    game_ids = []
    for gid, team_avgs in game_avgs.iteritems():
        for ta, feature_team_id in zip(team_avgs.iteritems(), ['-0', '-1']):
            tid, stats = ta
            for name, value in stats.iteritems():
                stat_key = TF_FEATURE_NAME(name + feature_team_id)
                if stat_key not in features:
                    features[stat_key] = []
                features[stat_key].append(value)
        labels.append(map(lambda tk: input_labels[gid][w][tk], team_avgs.keys()))
        game_ids.append(gid)

    return [features, labels, tuple(game_ids)]

def cast_to_nparray(**kwargs):
    """Casts input elements to nparrays.

    Args:
         lis: lis of inputs
    
    Returns: lis of nparray types
    """
    lis = kwargs['lis']
    
    for k in lis[0].keys():
        lis[0][k] = np.array(lis[0][k])

    lis[1] = np.array(lis[1])

    return lis

def z_score_labels(**kwargs):
    """zscores an input with shape [n, 2]. If a different shape
       is given, an error will be thrown. Average and standard 
       deviation are derived from the input by numpy.

    Args:
         labels: inputs to zscore
    
    Returns: zscored labels where each element is mapped 
             back to the same place
    Raises:
           ValueError: if input shape is not [n, 2]
    """
    labels = kwargs['labels']

    flat = reduce(lambda l1,l2: l1 + l2, labels)
    if len(flat) % 2 == 0:        
        zs = da.z_scores(data=flat)
        inds = [i for i in zip(range(0, len(flat) - 1, 2), range(1, len(flat), 2))]
        return map(lambda i: [zs[i[0]], zs[i[-1]]], inds)
    else:
        raise ValueError("Labels must have shape [n, 2]")

def histogram_games(**kwargs):
    """Creates a histogram of game ids by a given value. The corresponding
       key is used for lookups in game_infos.

    Args:
         game_infos: home, away etc. information for a game, see game.csv
         game_stats: set of game ids
         histo_key: key to use when performing lookups in game_infos
    
    Returns: 
    """
    game_infos, game_stats, histo_key = kwargs['game_infos'], kwargs['game_stats'], kwargs['histo_key']

    histo = {}
    for gid in game_stats:
        info = game_infos[gid]
        if info[histo_key] not in histo:
            histo[info[histo_key]] = []
        histo[info[histo_key]].append(gid)
    
    return histo

def stochastic_split_data(**kwargs):
    """Computes a randomized split of data by some percentage. Note, this
       function will produce different outputs for the same input as
       index generation is randomized. The split will tend to
       be slightly under the specified percentage due to rounding logic.

    Args:
         game_histo: histogram of games
         split_percentage: percentage to split games. The first
                           output will have split_percentage 
                           values.
    
    Returns: tuple of lists
    """
    gh, sp = kwargs['game_histo'], float(kwargs['split_percentage'])

    train = []
    test = []
    train_divi = True
    for k, games in gh.iteritems():
        count = len(games)
        if count == 1:
            if train_divi:
                train.append(games[0])
                train_divi = False
            else:
                test.append(games[0])
                train_divi = True
        elif count == 2:
            num = randint(0, 1)
            train.append(games[num])
            test.append(games[int(not num)])
        else:
            train_split = int(round(count * sp))
            test_split = count - train_split
            if test_split == 0:
                train_split = int(math.ceil(float(count) / 2))
            ind_range = set(range(count))
            train_ind = set(random.sample(ind_range, train_split))
            test_ind = ind_range.difference(train_ind)
            train += [games[e] for e in train_ind]
            test += [games[e] for e in test_ind]

    return train, test

def static_split_data(**kwargs):
    """Computes a deterministic split of data. Given the same input, this
       function will produce the same output. The split may be slightly
       different from the specified percentage due to rounding logic.

    Args:
         game_histo: histogram of games
         split_percentage: percentage to split games. The first
                           output will have split_percentage 
                           values.

    Returns: tuple of lists
    """
    gh, sp = kwargs['game_histo'], float(kwargs['split_percentage'])

    train = []
    test = []
    train_divi = True
    keys = list(gh.keys())
    keys.sort(key=lambda x: du.parse(x))
    for d in keys:
        count = len(gh[d])
        k = d
        if count == 1:
            if train_divi:
                train.append(gh[d][0])
                train_divi = False
            else:
                test.append(gh[d][0])
                train_divi = True
        elif count == 2:
            train.append(gh[d][0])
            test.append(gh[d][1])
        else:
            train_split = int(round(count * sp))
            test_split = count - train_split
            if test_split == 0:
                train_split = int(math.ceil(float(count) / 2))
            num_games = len(gh[d])
            ind_range = set(range(num_games))
            train_ind = set(range(int(math.floor(num_games * sp))))
            test_ind = ind_range.difference(train_ind)
            train += [gh[d][e] for e in train_ind]
            test += [gh[d][e] for e in test_ind] 

    train.sort()
    test.sort()
    return train, test

def stochastically_randomize_vector(**kwargs):
    """For a given rate, will randomize an input vector. The rate
       is used as a denominator, i.e. 1 / rate. So increases in rate 
       will decrease the likelihood of randomizing the input vector.

    Args:
         net: the current tensor
         rate: rate at which to randomize the input vector
    
    Returns: tensor
    """
    in_net, r = kwargs['net'], kwargs['rate']
    net = tf.map_fn(lambda gf: tf.cond(tf.equal(tf.constant(0), tf.random.uniform([1], 
                                       maxval=int(r), dtype=tf.int32))[0], 
                                       true_fn=lambda: tf.random_shuffle(gf), 
                                       false_fn=lambda: gf), in_net)
    return net

def stochastically_randomize_half_vector(**kwargs):
    """For a given rate, will randomize an half of an input vector. The rate
       is used as a denominator, i.e. 1 / rate. So increases in rate will
       decrease the likelihood of randomizing half of the input vector. The
       half that is randomized is stochastic with 50% chance of either occuring, 
       but not both.
       
    Args:
         net: the current tensor
         rate: rate at which to randomize the input vector
         ub: upperbound on the split for the input vector
    
    Returns: tensor
    """
    in_net, r, ub = kwargs['net'], kwargs['rate'], kwargs['ub']
    keep_range = tf.cond(tf.equal(tf.constant(0), tf.random.uniform([1], maxval=2, dtype=tf.int32))[0], 
                         true_fn=lambda: tf.range(0, ub / 2), 
                         false_fn=lambda: tf.range(ub / 2, ub))
    randomize_range = tf.sets.difference([tf.range(ub)], [keep_range]).values
    net = tf.map_fn(lambda gf: tf.cond(tf.equal(tf.constant(0), tf.random.uniform([1], maxval=r, dtype=tf.int32))[0], 
                                           true_fn=lambda: tf.gather(gf, tf.concat([keep_range, 
                                           tf.random_shuffle(randomize_range)], axis=0)), 
                                           false_fn=lambda: gf), in_net)
    return net

def randomize_vector(**kwargs):
    """Deterministically randomizes the whole input vector.

    Args:
         net: the current tensor
    
    Returns: tensor
    """
    in_net = kwargs['net']
    return tf.map_fn(lambda gf: tf.random_shuffle(gf), in_net)

def mean_power_error(**kwargs):
    """Computes loss across an input batch of elements by 
       1 / n * (mult(pow(abs(x - y), power), weight)). 

    Args:
         labels: actual outcomes
         logits: scores from network
         power: power to raise absolute value to
         weight: multiply result of power
    
    Returns: average loss for batch
    """
    labels, logits, p, w = kwargs['labels'], kwargs['logits'], kwargs['power'], kwargs['weight']

    return tf.math.reduce_mean(tf.map_fn(lambda i: tf.math.multiply(tf.math.pow(tf.math.abs(
        tf.math.subtract(tf.gather(logits, i), tf.gather(labels, i))), p), w), 
                                         tf.range(tf.shape(logits)[0]), dtype=tf.float32))

def mean_piecewise_power_error(**kwargs):
    """Allows for two power functions to be used. If input is less than alpha,
       input is raised to power_alpha. Otherwise the input is raised to power.
       Loss is computed by 1 / n * (mult(abs(x - y), (power | power_alpha)))

    Args:
         labels: actual outcomes
         logits: scores from network
         power_alpha: a power
         alpha: exclusive upper bound for using power_alpha
         power: a power
    
    Returns: average loss for batch
    """
    labels, logits, pa, a, p = kwargs['labels'], kwargs['logits'], kwargs['power_alpha'], kwargs['alpha'], kwargs['power']

    return tf.math.reduce_mean(tf.map_fn(lambda i: 
                                        tf.map_fn(lambda s: tf.cond(tf.math.less(s, a), 
                                        true_fn=lambda: tf.math.pow(s, pa), false_fn=lambda: tf.math.pow(s, p)), 
                                        tf.math.abs(tf.math.subtract(tf.gather(logits, i), 
                                        tf.gather(labels, i)))), tf.range(tf.shape(logits)[0]), dtype=tf.float32))

def mean_absolute_error(**kwargs):
    """Computes MAE on input batch.

    Args:
         labels: actual outcomes
         logits: scores from network
    
    Returns: average loss for batch
    """
    labels, logits = kwargs['labels'], kwargs['logits']

    return mean_power_error(labels=labels, logits=logits, power=1.0, weight=1.0)

RANDOMIZER = {'stochastically_randomize_vector': stochastically_randomize_vector, 
                       'stochastically_randomize_half_vector': stochastically_randomize_half_vector}
ACTIVATION = {'relu': tf.nn.relu, 'relu6': tf.nn.relu6, 'sigmoid': tf.math.sigmoid, 'leaky_relu': tf.nn.leaky_relu}
REGULARIZATION = {'l2': tf.contrib.layers.l2_regularizer, 'l1': tf.contrib.layers.l1_regularizer}
SPLIT_FUNCTION = {'stochastic': stochastic_split_data, 'static': static_split_data}
LOSS_FUNCTION = {'mean_power_error': mean_power_error, 'mean_absolute_error': mean_absolute_error, 
                 'mean_piecewise_power_error': mean_piecewise_power_error}

def model_fn(features, labels, mode, params):
    """Creates the model for all three estimator modes, namely, PREDICTION,
       TRAIN, and TEST. Configured by protobufs, see config directory.

    Args:
         features: example inputs
         labels: example outputs
         params: map of extra params
    
    Returns: estimator specification
    """
    ec, fc, da, rf, r, do, hl, n, reg, act, ty, ol, lr, t, sc, lf, p, pa, a, w = params['estimator_config'],\
                                                            params['feature_columns'], 'dataAugment', 'randomizerFunc',\
                                                            'rate', 'dropout', 'hiddenLayer', 'neurons', 'regularization',\
                                                            'activation', 'type', 'outputLayer', 'learningRate', 'train',\
                                                            'scale', 'lossFunction', 'power', 'powerAlpha', 'alpha', 'weight'
    hidden_layers = ec.get(hl)
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        ub = len(features)

        if ec.get(da):
            net = RANDOMIZER.get(ec.get(da).get(rf))(net=net, rate=ec.get(da).get(r), ub=ub)        
            net = tf.reshape(net, [-1, ub])

    tf.summary.histogram('input_layer', net)
    if ec.get(do):
        net = tf.nn.dropout(net, keep_prob=ec.get(do).get(r))

    #hidden layer
    for unit, num in zip(hidden_layers, range(len(hidden_layers))):
        layer_params = {}
        if unit.get(reg):
            layer_params.update({'kernel_regularizer': REGULARIZATION.get(unit.get(reg).get(ty))(unit.get(reg).get(sc))})

        net = tf.layers.dense(net, unit.get(n), **layer_params)
        tf.summary.histogram("weights_%s_%s" % (str(unit.get(n)), str(num)), net)

        if unit.get(act):
            net = ACTIVATION.get(unit.get(act).get(ty))(net, name=unit.get(act).get(ty) + str(unit.get(n)))
            tf.summary.histogram("activations_%s_%s" % (str(unit.get(n)), str(num)), net)

    #logits
    output_params = {}
    if ec.get(ol).get(reg):
        output_params.update({'kernel_regularizer': REGULARIZATION.get(ec.get(ol).get(reg).get(ty))(ec.get(ol).get(reg).get(sc))})
    if ec.get(ol).get(act):
        output_params.update({'activation': ACTIVATION.get(ec.get(ol).get(act).get(ty))})
    net = tf.layers.dense(net, ec.get(ol).get(n), **output_params)

    tf.summary.histogram('logits_' + str(2), net)

    predicted_winner = tf.argmax(net, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'scores': net,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = LOSS_FUNCTION[ec[lf][ty]](labels=tf.dtypes.cast(labels, tf.float32), logits=net, power=ec[lf].get(p), 
                                     power_alpha=ec[lf].get(pa), alpha=ec[lf].get(a), weight=ec[lf].get(w))
    accuracy = tf.metrics.accuracy(tf.argmax(labels, 1), predicted_winner)
    wl_acc = 'win loss accuracy'
    tf.summary.scalar(wl_acc, accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={wl_acc: accuracy})
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=ec.get(t).get(lr))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def train_input_fn(features, labels, batch_size):
    """Provides input to the model during training. Data is shuffled and 
       batched.

    Args:
         features: inputs
         labels: outputs
         batch_size: size of batches for network
    
    Returns: data
    """
    return tf.data.Dataset.from_tensor_slices((dict(features), labels))\
                             .shuffle(1000).repeat().batch(batch_size)
    
def eval_input_fn(features, labels, batch_size):
    """Provides input to the model during testing.

    Args:
         features: inputs
         labels: outputs
         batch_size: size of batches for network
    
    Returns: data
    """
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

def z_scores(**kwargs):
    """zscores values in given map.

    Args:
         data: map of keys to collections
    
    Returns: a map
    """
    fs = kwargs['data']
    
    return {f: da.z_scores(data=fs[f]) for f in fs.keys()}

def predict_input_fn(features, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((dict(features), labels)).batch(batch_size)

def run_model(**kwargs):
    """Creates train features, train labels, test features, and test labels 
       and executes model. The model is both trained and tested.

    Args:
         team_avgs: averages for teams
         split: input data split into train and test
         labels: outputs of examples
         features: inputs of examples
         estimator_config: protobuf.config file
    
    Returns: None
    """
    avgs, split, labels, feat, ec, t, scs, n, hl, md, ts, ets, bs, rd, tr, tst = kwargs['team_avgs'], kwargs['split'],\
                                                                kwargs['labels'], kwargs['features'],\
                                                                kwargs['estimator_config'], 'train',\
                                                                'saveCheckpointsSteps', 'neurons',\
                                                                'hiddenLayer', 'modelDir', 'trainSteps',\
                                                                'evalThrottleSecs', 'batchSize', 'run_dir',\
                                                                kwargs['train_data'], kwargs['test_data']
    #mock up
    predict = False
    saved_model = path.abspath('single_dist_cfbstats-com-2005-1-5-0_2a1429a8-b266-11e9-9055-b8975a6ac69a')

    train_features, train_labels = tr
    test_features, test_labels = tst

    feature_cols = []
    for f in feat:
        feature_cols.append(tf.feature_column.numeric_column(key=f))

    model_dir = saved_model if predict else path.join(ec[t][md], "%s_%s" % (str(ec[rd]), str(uuid.uuid1()))) 
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=ec[t][scs])
    model = tf.estimator.Estimator(model_fn=model_fn, 
                                        params={'feature_columns': feature_cols, 'estimator_config': ec}, 
                                        config=run_config, 
                                        model_dir=model_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_features, 
                                                                        train_labels, ec[t][bs]), 
                                        max_steps=ec[t][ts])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(test_features, 
                                                                     test_labels, ec[t][bs]), 
                                      throttle_secs=ec[t][ets])
    if predict:
        output = model.predict(lambda: predict_input_fn(test_features, test_labels, ec[t][bs]))
        c = 1
        try:
            while True:
                print("Num: %s, data: %s" % (str(c), str(next(output))))
                c += 1
        except StopIteration:
            print('Done iterating')
    else:
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

def season_dirs(**kwargs):
    """Each config specifies a pattern and this function resolves 
       that pattern against disk and associates the output with 
       the configuration file's name.

    Args:
         configs: protobuf.config configurations 
    
    Returns: tuple of map and set
    """
    cs, d, dp = kwargs['configs'], 'data', 'directoryPattern'
    
    result = {}
    for c in cs:
        result[c[0]] = glob.glob(c[1].get(d).get(dp))

    return result, set(reduce(lambda f1,f2: f1 + f2, result.values()))

def read_config(**kwargs):
    """Given a file, reads it in and maps its values to a protobuf object.
       The protobuf is then converted to a python dict and returned.

    Args:
         estimator_file: protobuf.config file
    
    Returns: map
    """
    ef = kwargs['estimator_file']

    pb = estimator_pb2.Estimator()
    result = {}
    with open(path.abspath(ef), 'rb') as fh:
        proto_str = fh.read()
        text_format.Merge(proto_str, pb)
        result = MessageToDict(pb)
    
    return result

def season_data(**kwargs):
    """Computes data by season. Reads in values from team-game-statistics.csv and 
       game.csv and dervies more values.

    Args:
         dirs: directories from which to read input
    
    Returns: map
    """
    ds = kwargs['dirs']

    result = {}
    for season_dir in ds:
        gs = game.game_stats(directory=season_dir)
        team_stats = tgs.team_game_stats(directory=season_dir)
        avgs, _ = averages(team_game_stats=team_stats, game_infos=gs, skip_fields=model.UNDECIDED_FIELDS)
        team_stats = {k: team_stats[k] for k in avgs.keys()}        
        labels = tgs.add_labels(team_game_stats=team_stats)
        histo = histogram_games(game_infos=gs, game_stats=avgs, histo_key='Date')   
        #features = da.normal_dists(field_avgs=regression_data(game_averages=avgs, labels=labels)[0]).keys()
        reg = regression_data(game_averages=avgs, labels=labels)
        features = da.normal_dists(field_avgs=reg[0])

        result.update({season_dir: {'features': features.keys(), 'labels': labels, 'team_avgs': avgs, 
                                    'game_stats': gs, 'team_stats': team_stats, 'histo': histo, 'regression_data': reg, 
                                    'norm_data': features}})

    return result

def split_model_data(**kwargs):
    split, md = kwargs['data_split'], kwargs['model_data']
    feat, lab, games = md
    train, test = split

    gid_to_ind = {v: i for i, v in enumerate(games)}
    train_inds, test_inds = map(lambda gids: map(lambda g: gid_to_ind[g], gids), (train, test))
    train_lab, test_lab = map(lambda inds: map(lambda i: lab[i], inds), (train_inds, test_inds))

    train_feat = {k: map(lambda i: feat[k][i], train_inds) for k in feat.keys()}
    test_feat = {k: map(lambda i: feat[k][i], test_inds) for k in feat.keys()}

    return ((train_feat, train_lab), (test_feat, test_lab))

def evaluate_models(**kwargs):
    """Evalutes models by taking in associated data and computing splits before executing the 
       model. So a stochastic split function would be executed twice for the same configuration.
     

    Args:
         file_configs: tuples of associated configs and directories 
         sea_dirs: directories by season
         all_sea_data: data across all seasons
    
    Returns: None
    """
    fcs, sd, rd, asd, sp, dk, h, ta, ls, fs, msd, sf, rps, nd, reg_d, ms = kwargs['file_configs'], kwargs['sea_dirs'], 'run_dir',\
                                                   kwargs['all_sea_data'], 'splitPercent', 'data', 'histo',\
                                                   'team_avgs', 'labels', 'features', 'model_sub_dir', 'splitFunction',\
                                                   'runsPerSeason', 'norm_data', 'regression_data', kwargs['model_splits']

    for f in fcs:
        ec, dirs = f
        for d in dirs:
            sea_data = asd[d]
            ec.update({rd: "%s_%s" % (ec[msd], path.basename(d))})
            split = ms[d][ec[dk][sf]][ec[dk][sp]]
            norm_feats, norm_labels = [z_scores(data={k: sea_data[reg_d][0][k] for k in sea_data[nd].keys()}), 
                                                           z_score_labels(labels=sea_data[reg_d][1])]
            train, test = split_model_data(data_split=split, model_data=(norm_feats, norm_labels, sea_data[reg_d][-1]))
            np_train, np_test = cast_to_nparray(lis=list(train)), cast_to_nparray(lis=list(test))
            for i in range(ec[rps]):
                run_model(team_avgs=sea_data[ta], split=split, labels=sea_data[ls], features=sea_data[fs], 
                          estimator_config=ec, train_data=np_train, test_data=np_test)

def model_data_splits(**kwargs):
    file_cs, sp, sf, dk, sea_data, h = kwargs['file_configs'], 'splitPercent', 'splitFunction', 'data', kwargs['season_data'], 'histo'

    splits = {}
    for fc in file_cs:
        fn, dirs, conf = fc
        for d in dirs:
            if d not in splits:
                splits[d] = {}
            if conf[dk][sf] not in splits[d]:
                splits[d][conf[dk][sf]] = {}
            if conf[dk][sp] not in splits[d][conf[dk][sf]]:
                splits[d][conf[dk][sf]][conf[dk][sp]] = SPLIT_FUNCTION[conf[dk][sf]](game_histo=sea_data[d][h], 
                                                                                     split_percentage=conf[dk][sp])
   
    return splits

def main(args):
    parser = argparse.ArgumentParser(description='Predict scores of college football games')
    parser.add_argument('--estimator_configs', nargs='+', required=True, help='List of model configs')
    args = parser.parse_args() 
    cf = 'config'
    dc = '.' + cf
  
    valid_files = filter(lambda f: f.endswith(dc), args.estimator_configs)
    if not valid_files:
        print("--estimator_configs each file must end with %s to be processed" % (str(dc)))
    else:
        file_configs = map(lambda f: (f, path.basename(f).replace(dc, ''), read_config(estimator_file=f)), valid_files)
        for fc in file_configs:
            fc[-1].get(cf).update({'model_sub_dir': fc[1]})

        sea_dirs, all_dirs = season_dirs(configs=map(lambda c: (c[0], c[-1][cf]), file_configs))
        print("Reading data from these directories: ")
        util.print_collection(coll=all_dirs)
        sea_data, cs = util.model_data(cache_dir=DATA_CACHE_DIR, cache_read_func=util.read_from_cache, 
                                   cache_write_func=util.write_to_cache, all_dirs=all_dirs, comp_func=season_data, 
                                   comp_func_args={}, context_dir=path.dirname(list(all_dirs)[0]))
        util.print_cache_reads(coll=cs, data_origin=DATA_CACHE_DIR)
        splits = model_data_splits(file_configs=map(lambda f: (f[0], sea_dirs[f[0]], f[-1][cf]), file_configs), season_data=sea_data)
        evaluate_models(file_configs=map(lambda f: (f[-1][cf], sea_dirs[f[0]]), file_configs), sea_dirs=sea_dirs, 
                        all_sea_data=sea_data, model_splits=splits)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    
