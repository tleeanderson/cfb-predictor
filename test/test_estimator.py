import estimator as est
import constants as const
import numpy as np

def test_averages():
    gid_1, gid_2, gid_3, tid_1, tid_2, ps, d, sf = 'gid1', 'gid2', 'gid3', 'tid1', 'tid2', 'Points',\
                                               'Date', 'skip_field'
    vt, ht, sep, octo, nov, t1_val, t2_val = const.VISIT_TEAM_CODE, const.HOME_TEAM_CODE, '09/01/2005',\
                                             '10/01/2005', '11/01/2005', 5, 6
    
    tgs = {gid_1: {tid_1: {ps: t1_val, sf: 1}, tid_2: {ps: t2_val, sf: 1}},
    gid_2: {tid_1: {ps: t1_val, sf: 1}, tid_2: {ps: t2_val, sf: 1}},
    gid_3: {tid_1: {ps: t1_val, sf: 1}, tid_2: {ps: t2_val, sf: 1}}}
    game_infos = {gid_1: {vt: tid_1, ht: tid_2, d: sep},
                  gid_2: {vt: tid_1, ht: tid_2, d: octo},
                  gid_3: {vt: tid_1, ht: tid_2, d: nov}}

    out, na = est.averages(team_game_stats=tgs, game_infos=game_infos, skip_fields={sf})
    t1_f = float(t1_val)
    t2_f = float(t2_val)

    assert({gid_2, gid_3} == set(out.keys()))
    assert({tid_1, tid_2} == set(out[gid_2].keys()))
    assert({tid_1, tid_2} == set(out[gid_3].keys()))
    assert([t1_f, t1_f] == [out[gid_2][tid_1][ps], out[gid_3][tid_1][ps]])
    assert([t2_f, t2_f] == [out[gid_2][tid_2][ps], out[gid_3][tid_2][ps]])
    assert(all([sf not in tids for tids in [out[gid_2], out[gid_3]]]))
    assert({gid_1} == set(na))

def test_regression_data():
    gid_1, tid_1, tid_2, ps, w, ry = 'gid1', 'tid1', 'tid2', 'Points', 'Winner', 'Rush-Yard'
    gid_2, ps1, ps2, ps3, ps4, ry1, ry2, ry3, ry4 = 'gid2', 12, 20, 32, 44, 180, 200, 220, 240
    p1, p2, p3, p4 = 1, 2, 3, 4
    ga = {gid_1: {tid_1: {ps: ps1, ry: ry1}, tid_2: {ps: ps2, ry: ry2}}, 
          gid_2: {tid_1: {ps: ps3, ry: ry3}, tid_2: {ps: ps4, ry: ry4}}}
    labels = {gid_1: {w: {tid_1: p1, tid_2: p2}}, 
              gid_2: {w: {tid_1: p3, tid_2: p4}}}

    out_feat, out_lab, gids = est.regression_data(game_averages=ga, labels=labels)

    games_by_row = np.transpose([out_feat[f] for f in out_feat])
    assert({ry + '-0', ry + '-1', ps + '-0', ps + '-1'} == set(out_feat.keys()))
    assert(any([{ps1, ry1, ps2, ry2} == set(gr) for gr in games_by_row]))
    assert(any([{ps3, ry3, ps4, ry4} == set(gr) for gr in games_by_row]))
    games_by_row_sets = tuple([set(gr) for gr in games_by_row])
    out_lab_sets = tuple([set(la) for la in out_lab])
    assert(games_by_row_sets.index({ps1, ry1, ps2, ry2}) == out_lab_sets.index({p1, p2}))
    assert(games_by_row_sets.index({ps3, ry3, ps4, ry4}) == out_lab_sets.index({p3, p4}))
    assert(games_by_row_sets.index({ps1, ry1, ps2, ry2}) == out_lab_sets.index({p1, p2}) 
           == gids.index(gid_1))
    assert(games_by_row_sets.index({ps3, ry3, ps4, ry4}) == out_lab_sets.index({p3, p4}) 
           == gids.index(gid_2))

def test_zscore_labels():
    labels = [[i, i + 1] for i in range(10)]
    bad_labels = [[1, 2], [3], [4, 5]]
    lab_avg, lab_std = np.average(labels), np.std(labels)

    out, out_avg, out_std = est.z_score_labels(labels=labels)
    
    assert({2} == set([len(ls) for ls in labels]))
    assert(lab_avg == out_avg)
    assert(lab_std == out_std)
    try:
        _ = est.z_score_labels(labels=bad_labels)
    except ValueError:
        assert(True)
    else:
        assert(False)

def test_histogram_games():
    gid_1, gid_2, gid_3, d = 'gid1', 'gid2', 'gid3', 'Date'
    sep, octo, nov = '09/01/2005', '10/01/2005', '11/01/2005'
    gs = {gid_1, gid_2, gid_3}
    game_infos = {gid_1: {d: sep}, gid_2: {d: octo}, gid_3: {d: nov}}

    out = est.histogram_games(game_infos=game_infos, game_stats=gs, histo_key=d)

    assert({sep, octo, nov} == set(out.keys()))
    assert({gid_1} == set(out[sep]))
    assert({gid_2} == set(out[octo]))
    assert({gid_3} == set(out[nov]))

def _test_split(**kwargs):
    train, test, et = kwargs['train'], kwargs['test'], kwargs['exp_total']

    assert(et == len(train) + len(test))
    assert(set() == set(train).intersection(set(test)))

def test_stochastic_split_data():
    sep, octo, nov = '09/01/2005', '10/01/2005', '11/01/2005'
    game_histo = {sep: list(range(10)), octo: list(range(10, 20)), nov: list(range(20, 30))}
    exp_total = sum([len(game_histo[r]) for r in game_histo])

    train, test = est.stochastic_split_data(game_histo=game_histo, split_percentage=0.5)

    _test_split(train=train, test=test, exp_total=exp_total)

def test_static_split_data():
    sep, octo, nov = '09/01/2005', '10/01/2005', '11/01/2005'
    game_histo = {sep: list(range(10)), octo: list(range(10, 20)), nov: list(range(20, 30))}
    exp_total = sum([len(game_histo[r]) for r in game_histo])

    train, test = est.static_split_data(game_histo=game_histo, split_percentage=0.5)
    train_2, test_2 = est.static_split_data(game_histo=game_histo, split_percentage=0.5)
    
    _test_split(train=train, test=test, exp_total=exp_total)
    _test_split(train=train_2, test=test_2, exp_total=exp_total)
    assert(train == train_2)
    assert(test == test_2)

def test_split_model_data():
    gid_1, gid_2, gid_3, gid_4, f1, f2 = 'gid1', 'gid2', 'gid3', 'gid4', 'f1', 'f2'
    sp1, sp2, l1, l2, l3, l4, l5, l6, l7, l8 = [gid_1, gid_2], [gid_3, gid_4], 1, 2, 3, 4, 5, 6, 7, 8
    feat, lab = {f1: list(range(4)), f2: list(range(4))}, [[l1, l2], [l3, l4], [l5, l6], [l7, l8]] 
    games = [gid_1, gid_2, gid_3, gid_4]

    t1, t2 = est.split_model_data(data_split=(sp1, sp2), model_data=(feat, lab, games))
    
    assert(({f1: list(range(2)), f2: list(range(2))}, [[l1, l2], [l3, l4]], sp1) == (t1[0], t1[1], t1[2]))
    assert(({f1: list(range(2, 4)), f2: list(range(2, 4))}, [[l5, l6], [l7, l8]], sp2) == (t2[0], t2[1], t2[2]))

def test_compare_pred_scores():
    pk, ak, dk, ck = 'predictions', 'actual', 'distance', 'correct'
    key_args = {'pred_key': pk, 'actual_key': ak , 'distance_key': dk, 'correct_key': ck}
    s1, s2, s3, s4, gid_1, gid_2, l1, l2, l3, l4 = 1, 2, 3, 4, 'gid1', 'gid2', 5, 6, 7, 8
    pred_scores, gids, ol = [[s1, s2], [s3, s4]], [gid_1, gid_2], [[l1, l2], [l4, l3]]
 
    out = est.compare_pred_scores(pred_scores=pred_scores, gids=gids, 
                            original_labels=ol, **key_args)

    assert({gid_1, gid_2} == set(out.keys()))
    assert({pk: pred_scores[0], ak: ol[0], dk: [4, 4], ck: True} == out[gid_1])
    assert({pk: pred_scores[1], ak: ol[1], dk: [5, 3], ck: False} == out[gid_2])

def test_prediction_summary():
    dk, ck = 'distance', 'correct'
    key_args = {'distance_key': dk, 'correct_key': ck}
    stddev, mean, gid_1, gid_2 = 1.0, 4.5, 'gid1', 'gid2'
    pred_comparisons = {gid_1: {ck: True, dk: [1, 1]}, gid_2: {ck: False, dk: [1, 1]}}
    out_std, out_m, out_np = 'stddev_of_points', 'mean_of_points', 'num_predictions'
    out_pc, out_c, out_ic, out_adbt = 'percent_correct', 'correct', 'incorrect',\
                                      'average_distance_by_team'
                             
    out = est.prediction_summary(pred_comparisons=pred_comparisons, distance_key=dk, 
                                 correct_key=ck, stddev=stddev, mean=mean)

    assert(stddev == out[out_std])
    assert(mean == out[out_m])
    assert(2 == out[out_np])
    assert(0.5 == out[out_pc])
    assert(1 == out[out_c])
    assert(1 == out[out_ic])
    assert(1 == out[out_adbt])

def test_model_data_splits():
    d1, d2, stoch, static, sp1 = 'd1', 'd2', 'stochastic', 'static', 0.78
    sep, octo, nov, dk, sfk, spk, h = '09/01/2005', '10/01/2005', '11/01/2005', 'data', 'splitFunction',\
                                   'splitPercent', 'histo'
    game_histo = {sep: list(range(10)), octo: list(range(10, 20)), nov: list(range(20, 30))}
    sea_data = {d1: {h: game_histo}, d2: {h: game_histo}}
    exp_total = sum([len(game_histo[r]) for r in game_histo])
    file_configs = [('', [d1, d2], {dk: {sfk: stoch, spk: sp1}}), 
                    ('', [d1, d2], {dk: {sfk: static, spk: sp1}})]

    out = est.model_data_splits(file_configs=file_configs, season_data=sea_data)    

    assert({d1, d2} == set(out.keys()))
    assert({stoch, static} == set(np.reshape([list(out[d].keys()) for d in out], [-1])))
    assert({sp1} == set(np.reshape([[list(out[d][sf].keys()) for sf in out[d]] for d in out], [-1])))
    for d, dv in out.items():
        for sf, sfv in dv.items():
            for sp, split in sfv.items():
                _test_split(train=split[0], test=split[1], exp_total=exp_total)    
