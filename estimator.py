import main as temp_lib
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

TF_FEATURE_NAME = lambda f: f.replace(' ', '-')
BATCH_SIZE = 20
TRAIN_STEPS = 3000
DATA_CACHE_DIR = 'data_cache'

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

def binary_classification_data(**kwargs):
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
    game_avgs, input_labels, ps, w = kwargs['game_averages'], kwargs['labels'], 'Points', 'Winner'

    features = {}
    labels = []
    for gid, team_avgs in game_avgs.iteritems():
        for ta, feature_team_id in zip(team_avgs.iteritems(), ['-0', '-1']):
            tid, stats = ta
            for name, value in stats.iteritems():
                stat_key = TF_FEATURE_NAME(name + feature_team_id)
                if stat_key not in features:
                    features[stat_key] = []
                features[stat_key].append(value)
        labels.append(map(lambda tk: input_labels[gid][w][tk], team_avgs.keys()))

    for k in features.keys():
        features[k] = np.array(features[k])

    flat = reduce(lambda l1,l2: l1 + l2, labels)
    zs = da.z_scores(data=flat)
    inds = [i for i in zip(range(0, len(flat) - 1, 2), range(1, len(flat), 2))]
    scores = map(lambda i: [zs[i[0]], zs[i[-1]]], inds)

    return features, np.array(scores)

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

def stochastic_split_data(**kwargs):
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

def static_split_data(**kwargs):
    gh, sp, shc = kwargs['game_histo'], float(kwargs['split_percentage']), kwargs['histo_count']

    train = []
    test = []
    train_divi = True
    keys = list(gh.keys())
    keys.sort(key=lambda x: du.parse(x))
    for d in keys:
        count = shc[d]
        k = d
        if count == 1:
            if train_divi:
                train.append(gh[k][0])
                train_divi = False
            else:
                test.append(gh[k][0])
                train_divi = True
        elif count == 2:
            train.append(gh[k][0])
            test.append(gh[k][1])
        else:
            train_split = int(round(count * sp))
            test_split = count - train_split
            if test_split == 0:
                train_split = int(math.ceil(float(count) / 2))
            num_games = len(gh[k])
            ind_range = set(range(num_games))
            train_ind = set(range(int(math.floor(num_games * sp))))
            test_ind = ind_range.difference(train_ind)
            train += [gh[k][e] for e in train_ind]
            test += [gh[k][e] for e in test_ind] 

    train.sort()
    test.sort()
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

def stochastically_randomize_vector(**kwargs):
    in_net, r = kwargs['net'], kwargs['rate']
    net = tf.map_fn(lambda gf: tf.cond(tf.equal(tf.constant(0), tf.random.uniform([1], 
                                       maxval=int(r), dtype=tf.int32))[0], 
                                       true_fn=lambda: tf.random_shuffle(gf), 
                                       false_fn=lambda: gf), in_net)
    return net

def stochastically_randomize_half_vector(**kwargs):
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
    in_net = kwargs['net']
    return tf.map_fn(lambda gf: tf.random_shuffle(gf), in_net)

def mean_power_error(**kwargs):
    labels, logits, p, w = kwargs['labels'], kwargs['logits'], kwargs['power'], kwargs['weight']

    return tf.math.reduce_mean(tf.map_fn(lambda i: tf.math.multiply(tf.math.pow(tf.math.abs(
        tf.math.subtract(tf.gather(logits, i), tf.gather(labels, i))), p), w), 
                                         tf.range(tf.shape(logits)[0]), dtype=tf.float32))

def mean_piecewise_power_error(**kwargs):
    labels, logits, pa, a, p = kwargs['labels'], kwargs['logits'], kwargs['power_alpha'], kwargs['alpha'], kwargs['power']

    return tf.math.reduce_mean(tf.map_fn(lambda i: 
                                        tf.map_fn(lambda s: tf.cond(tf.math.less(s, a), 
                                        true_fn=lambda: tf.math.pow(s, pa), false_fn=lambda: tf.math.pow(s, p)), 
                                        tf.math.abs(tf.math.subtract(tf.gather(logits, i), 
                                        tf.gather(labels, i)))), tf.range(tf.shape(logits)[0]), dtype=tf.float32))

def mean_absolute_error(**kwargs):
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

    predicted_winners = tf.argmax(net, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_winners[:, tf.newaxis],
            'probabilities': net,
            'logits': net,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = LOSS_FUNCTION[ec[lf][ty]](labels=tf.dtypes.cast(labels, tf.float32), logits=net, power=ec[lf].get(p), 
                                     power_alpha=ec[lf].get(pa), alpha=ec[lf].get(a), weight=ec[lf].get(w))
    accuracy = tf.metrics.accuracy(tf.argmax(labels, 1), predicted_winners)
    wl_acc = 'win loss accuracy'
    tf.summary.scalar(wl_acc, accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={wl_acc: accuracy})
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=ec.get(t).get(lr))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def train_input_fn(features, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((dict(features), labels))\
                             .shuffle(1000).repeat().batch(batch_size)
    
def eval_input_fn(features, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

def z_scores(**kwargs):
    fs = kwargs['data']
    
    return {f: da.z_scores(data=fs[f]) for f in fs.keys()}

def print_scores(**kwargs):
    scores = kwargs['scores']

    for s, data in scores.iteritems():
        mi = min(data)
        mx = max(data)
        print((s, mi, mx, mx - mi))
        
    raw_input()

def run_model(**kwargs):
    avgs, split, labels, feat, ec, t, scs, n, hl, md, ts, ets, bs, rd = kwargs['team_avgs'], kwargs['split'],\
                                                                kwargs['labels'], kwargs['features'],\
                                                                kwargs['estimator_config'], 'train',\
                                                                'saveCheckpointsSteps', 'neurons',\
                                                                'hiddenLayer', 'modelDir', 'trainSteps',\
                                                                'evalThrottleSecs', 'batchSize', 'run_dir'

    train_features, train_labels = regression_data(game_averages={gid: avgs[gid] for gid in split[0]}, 
                                              labels=labels)

    train_features = z_scores(data=train_features)
    train_features = {tf: train_features[tf] for tf in feat}

    test_features, test_labels = regression_data(game_averages={gid: avgs[gid] for gid in split[1]}, 
                                            labels=labels)


    test_features = z_scores(data=test_features)
    test_features = {tf: test_features[tf] for tf in feat}

    feature_cols = []
    for f in feat:
        feature_cols.append(tf.feature_column.numeric_column(key=f))      

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=ec[t][scs])
    classifier = tf.estimator.Estimator(model_fn=model_fn, 
                                        params={'feature_columns': feature_cols, 'estimator_config': ec}, 
                                        config=run_config, 
                                        model_dir=path.join(ec[t][md], "%s_%s" % (str(ec[rd]), 
                                                                                  str(uuid.uuid1()))))
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_features, 
                                                                        train_labels, ec[t][bs]), 
                                        max_steps=ec[t][ts])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(test_features, 
                                                                     test_labels, ec[t][bs]), 
                                      throttle_secs=ec[t][ets])
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

def season_dirs(**kwargs):
    cs, d, dp = kwargs['configs'], 'data', 'directoryPattern'
    
    result = {}
    for c in cs:
        result[c[0]] = glob.glob(c[1].get(d).get(dp))

    return result, set(reduce(lambda f1,f2: f1 + f2, result.values()))

def read_config(**kwargs):
    ef = kwargs['estimator_file']

    pb = estimator_pb2.Estimator()
    result = {}
    with open(path.abspath(ef), 'rb') as fh:
        proto_str = fh.read()
        text_format.Merge(proto_str, pb)
        result = MessageToDict(pb)
    
    return result

def season_data(**kwargs):
    ds = kwargs['dirs']

    result = {}
    for season_dir in ds:
        gs = temp_lib.game_stats(directory=season_dir)
        team_stats = temp_lib.team_game_stats(directory=season_dir)
        avgs = averages(team_game_stats=team_stats, game_infos=gs, skip_fields=model.UNDECIDED_FIELDS)
        team_stats = {k: team_stats[k] for k in avgs.keys()}        
        labels = tgs.add_labels(team_game_stats=team_stats)
        histo = histogram_games(game_infos=gs, game_stats=avgs, histo_key='Date')   
        features = da.normal_dists(field_avgs=regression_data(game_averages=avgs, labels=labels)[0]).keys()

        result.update({season_dir: {'features': features, 'labels': labels, 'team_avgs': avgs, 
                                    'game_stats': gs, 'team_stats': team_stats, 'histo': histo}})

    return result

def write_to_cache(**kwargs):
    cd, data = kwargs['cache_dir'], kwargs['data']

    for sea, d in data.iteritems():
        with open(path.join(path.abspath(cd), path.basename(sea)), 'wb') as fh:
            pickle.dump(d, fh)

def read_from_cache(**kwargs):
    cd, context = kwargs['cache_dir'], kwargs['context_dir']

    result = {}
    for sea in glob.glob(path.join(path.abspath(cd), '*')):
        with open(sea, 'rb') as fh:
            entry = path.join(context, path.basename(sea))
            result[entry] = pickle.load(fh)
            
    return result

def model_data(**kwargs):
    cd, ds, crf, cwf = kwargs['cache_dir'], kwargs['dirs'], kwargs['cache_read_func'], kwargs['cache_write_func']

    if not path.exists(cd):
        os.mkdir(cd)
    cache = crf(cache_dir=cd, context_dir=path.dirname(list(ds)[0]))
    compute_seasons = set(ds).difference(set(cache.keys()))
    data = season_data(dirs=compute_seasons)
    cwf(cache_dir=cd, data=data)

    return {k: data[k] if k in data else cache[k] for k in set(cache.keys()).union(set(data.keys()))}

def evaluate_models(**kwargs):
    fcs, sd, rd, asd, sp, dk, h, ta, ls, fs, msd, sf, rps = kwargs['file_configs'], kwargs['sea_dirs'], 'run_dir',\
                                                   kwargs['all_sea_data'], 'splitPercent', 'data', 'histo',\
                                                   'team_avgs', 'labels', 'features', 'model_sub_dir', 'splitFunction',\
                                                   'runsPerSeason'

    for f in fcs:
        ec, dirs = f
        for d in dirs:
            sea_data = asd[d]
            ec.update({rd: "%s_%s" % (ec[msd], path.basename(d))})
            split = SPLIT_FUNCTION[ec[dk][sf]](game_histo=sea_data[h], split_percentage=ec[dk][sp],
                                               histo_count={k: len(sea_data[h][k]) for k in sea_data[h].keys()})
            for i in range(ec[rps]):
                run_model(team_avgs=sea_data[ta], split=split, labels=sea_data[ls], features=sea_data[fs], 
                          estimator_config=ec)

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
        print("Reading in data from disk from these directories: %s" % (str(all_dirs)))
        sea_data = model_data(cache_dir=DATA_CACHE_DIR, dirs=all_dirs, cache_read_func=read_from_cache, 
                              cache_write_func=write_to_cache)
        print("Done reading data from disk")
        evaluate_models(file_configs=map(lambda f: (f[-1][cf], sea_dirs[f[0]]), file_configs), sea_dirs=sea_dirs, 
                        all_sea_data=sea_data)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    
