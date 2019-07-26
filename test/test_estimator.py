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
    assert(all(map(lambda tids: sf not in tids, [out[gid_2], out[gid_3]])))
    assert({gid_1} == set(na))

def test_regression_data():
    gid_1, tid_1, tid_2, ps, w, ry = 'gid1', 'tid1', 'tid2', 'Points', 'Winner', 'Rush-Yard'
    gid_2, ps1, ps2, ps3, ps4, ry1, ry2, ry3, ry4 = 'gid2', 12, 20, 32, 44, 180, 200, 220, 240
    p1, p2, p3, p4 = 1, 2, 3, 4
    ga = {gid_1: {tid_1: {ps: ps1, ry: ry1}, tid_2: {ps: ps2, ry: ry2}}, 
          gid_2: {tid_1: {ps: ps3, ry: ry3}, tid_2: {ps: ps4, ry: ry4}}}
    labels = {gid_1: {w: {tid_1: p1, tid_2: p2}}, 
              gid_2: {w: {tid_1: p3, tid_2: p4}}}

    out_feat, out_lab = est.regression_data(game_averages=ga, labels=labels)

    games_by_row = np.transpose(map(lambda f: out_feat[f], out_feat))
    assert({ry + '-0', ry + '-1', ps + '-0', ps + '-1'} == set(out_feat.keys()))
    assert(any(map(lambda gr: {ps1, ry1, ps2, ry2} == set(gr), games_by_row)))
    assert(any(map(lambda gr: {ps3, ry3, ps4, ry4} == set(gr), games_by_row)))
    games_by_row_sets = tuple(map(lambda gr: set(gr), games_by_row))
    out_lab_sets = tuple(map(lambda la: set(la), out_lab))
    assert(games_by_row_sets.index({ps1, ry1, ps2, ry2}) == out_lab_sets.index({p1, p2}))
    assert(games_by_row_sets.index({ps3, ry3, ps4, ry4}) == out_lab_sets.index({p3, p4}))

def test_zscore_labels():
    labels = [[i, i + 1] for i in range(10)]
    bad_labels = [[1, 2], [3], [4, 5]]

    out = est.z_score_labels(labels=labels)
    
    assert({2} == set(map(lambda ls: len(ls), labels)))
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
    game_histo = {sep: range(10), octo: range(10, 20), nov: range(20, 30)}
    exp_total = sum(map(lambda r: len(game_histo[r]), game_histo))

    train, test = est.stochastic_split_data(game_histo=game_histo, split_percentage=0.5)

    _test_split(train=train, test=test, exp_total=exp_total)

def test_static_split_data():
    sep, octo, nov = '09/01/2005', '10/01/2005', '11/01/2005'
    game_histo = {sep: range(10), octo: range(10, 20), nov: range(20, 30)}
    exp_total = sum(map(lambda r: len(game_histo[r]), game_histo))

    train, test = est.static_split_data(game_histo=game_histo, split_percentage=0.5)
    train_2, test_2 = est.static_split_data(game_histo=game_histo, split_percentage=0.5)
    
    _test_split(train=train, test=test, exp_total=exp_total)
    _test_split(train=train_2, test=test_2, exp_total=exp_total)
    assert(train == train_2)
    assert(test == test_2)
