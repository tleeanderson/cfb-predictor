import model
import test.utilities as tu

def test_attr_compare():
    v1, v2 = 1, 2

    out_max = model.attr_compare(v1=v1, v2=v2, compare=max)
    out_min = model.attr_compare(v1=v1, v2=v2, compare=min)

    assert(out_max == 1)
    assert(out_min == 0)

def test_evaluation():
    ps = 'Points'
    sm1, s1k, sm2, s2k, fw_mx, fw_mi, t = {ps: 5}, 's1k', {ps: 6}, 's2k',\
                                       {(ps): model.MAX_COMPARE},\
                                       {(ps): model.MIN_COMPARE}, 'tie'

    out_max = model.evaluation(stat_map1=sm1, stat_map2=sm2, st1_key=s1k, st2_key=s2k, 
                               field_win=fw_mx, undec_fields={})
    out_min = model.evaluation(stat_map1=sm1, stat_map2=sm2, st1_key=s1k, st2_key=s2k, 
                               field_win=fw_mi, undec_fields={})
    out_tie = model.evaluation(stat_map1=sm1, stat_map2=sm1, st1_key=s1k, st2_key=s2k, 
                               field_win=fw_mi, undec_fields={})

    assert(out_max[s1k] == 0)
    assert(out_max[s2k] == 1)
    assert(out_min[s1k] == 1)
    assert(out_min[s2k] == 0)
    assert(out_min[t] == False)
    assert(out_max[t] == False)
    assert(out_tie[s1k] == 0)
    assert(out_tie[s2k] == 0)
    assert(out_tie[t] == True)

def test_predict():
    tid_1, tid_2, ef, ps, v1, v2, w, t = '5', '6', model.evaluation, 'Points', 5, 6,\
                                      tu.WINNER, 'tie'
    ta = {tid_1: {ps: v1}, tid_2: {ps: v2}}

    out = model.predict(team_avgs=ta, game_code_id='1', tg_stats={}, eval_func=ef)
    
    assert(out[w] == tid_2)
    assert(out[t] == False)

    try:
        bad_ta = {k: ta[k] for k in ta.keys()}
        bad_ta.update({'8': {ps: v1}})
        model.predict(team_avgs=bad_ta, game_code_id='1', tg_stats={}, eval_func=ef)
    except ValueError:
        assert(True)
    else:
        assert(False)

def test_team_avgs():
    visit_team, home_team, v_tid, h_tid, gid = tu.VISIT_TEAM_CODE, tu.HOME_TEAM_CODE, '12', '22', str(9)
    sep, octo, nov, d, ps, v1, v2 = '09/01/2005', '10/01/2005', '11/01/2005', 'Date', 'Points', 2, 3
    gid_2, gid_3, gid_4, gid_5 = '2', '3', '4', '5'

    games = {gid: {visit_team: v_tid, home_team: h_tid, d: nov}, 
             str(1): {visit_team: '-1', home_team: '-1', d: sep}, 
             gid_2: {visit_team: h_tid, home_team: '-1', d: sep},
             gid_3: {visit_team: '-1', home_team: h_tid, d: sep},
             gid_4: {visit_team: '-1', home_team: v_tid, d: octo}, 
             gid_5: {visit_team: v_tid, home_team: '-1', d: octo}}
    stats = {gid_2: {h_tid: {ps: v1}}, gid_3: {h_tid: {ps: v1}}, gid_4: {v_tid: {ps: v2}}, 
             gid_5: {v_tid: {ps: v2}}}
    
    out = model.team_avgs(game_code_id=gid, game_data=games, tg_stats=stats)

    assert({v_tid, h_tid} == set(out.keys()))
    assert(v1 == out[h_tid][ps])
    assert(v2 == out[v_tid][ps])

def test_predict_all():
    gid_1, gid_2, gid_3, tid_1, tid_2, ps, v1, v2, d = 'gid1', 'gid2', 'gid3', 'tid1', 'tid2',\
                                                       'Points', 5, 10, 'Date'
    sep, octo, nov, npk = '09/01/2005', '10/01/2005', '11/01/2005', 'no_pred'

    visit_team, home_team, w = tu.VISIT_TEAM_CODE, tu.HOME_TEAM_CODE, tu.WINNER
    tgs = {gid_1: {tid_1: {ps: v1}, tid_2: {ps: v2}}, 
           gid_2: {tid_1: {ps: v1}, tid_2: {ps: v2}}, 
           gid_3: {tid_1: {ps: v1}, tid_2: {ps: v2}}}
    
    gi = {gid_1: {visit_team: tid_1, home_team: tid_2, d: nov}, 
          gid_2: {visit_team: tid_1, home_team: tid_2, d: octo}, 
          gid_3: {visit_team: tid_1, home_team: tid_2, d: sep}}

    out = model.predict_all(team_game_stats=tgs, game_infos=gi, no_pred_key=npk)

    assert({gid_3} == set(out[npk]))
    assert(tid_2 == out[gid_1][w])
    assert(tid_2 == out[gid_2][w])

def test_accuracy():
    ck, ik, tk, ak, w, tc, tie = 'correct', 'incorr', 'total', 'accuracy', tu.WINNER,\
                                 tu.TEAM_CODE, 'tie'
    gid_1, gid_2, gid_3, gid_4, tid_1, tid_2 = 'gid1', 'gid2', 'gid3', 'gid4', 'tid1', 'tid2'

    tgs = {gid_1: {w: {tc: tid_1}}, 
           gid_2: {w: {tc: tid_2}}, 
           gid_3: {w: {tc: tid_1}}}
    preds = {gid_1: {w: tid_1, tie: False}, gid_2: {w: tid_2, tie: False}, 
             gid_3: {w: tid_2, tie: False}}

    out = model.accuracy(tg_stats=tgs, predictions=preds, correct_key=ck, incorrect_key=ik, 
                         total_key=tk, skip_keys={}, acc_key=ak)

    assert(2 == out[ck])
    assert(1 == out[ik])
    assert(3 == out[tk])
    assert(2 / 3.0 == out[ak])

def test_accuracy_by_date():
    ck, ik, tk, ak, w, tc, tie, d, sep, octo, nov = 'correct', 'incorr', 'total', 'accuracy',\
                                                    tu.WINNER, tu.TEAM_CODE, 'tie', 'Date',\
                                                    '09/01/2005', '10/01/2005', '11/01/2005'
    gid_1, gid_2, gid_3, gid_4, gid_5, gid_6, tid_1, tid_2 = 'gid1', 'gid2', 'gid3', 'gid4', 'gid5', 'gid6', 'tid1', 'tid2'
    visit_team, home_team = tu.VISIT_TEAM_CODE, tu.HOME_TEAM_CODE

    tgs = {gid_1: {w: {tc: tid_1}}, 
           gid_2: {w: {tc: tid_2}}, 
           gid_3: {w: {tc: tid_1}}, 
           gid_4: {w: {tc: tid_2}}, 
           gid_5: {w: {tc: tid_1}}, 
           gid_6: {w: {tc: tid_2}}}
    preds = {gid_1: {w: tid_1, tie: False}, 
             gid_2: {w: tid_2, tie: False}, 
             gid_3: {w: tid_2, tie: False}, 
             gid_4: {w: tid_1, tie: False}, 
             gid_5: {w: tid_1, tie: False}, 
             gid_6: {w: tid_1, tie: False}}
    gi = {gid_1: {visit_team: tid_1, home_team: tid_2, d: nov},
          gid_2: {visit_team: tid_1, home_team: tid_2, d: nov},
          gid_3: {visit_team: tid_1, home_team: tid_2, d: octo},
          gid_4: {visit_team: tid_1, home_team: tid_2, d: octo},
          gid_5: {visit_team: tid_1, home_team: tid_2, d: sep}, 
          gid_6: {visit_team: tid_1, home_team: tid_2, d: sep}}

    out = model.accuracy_by_date(tg_stats=tgs, predictions=preds, game_info=gi, correct_key=ck, incorrect_key=ik, 
                         total_key=tk, skip_keys={}, acc_key=ak)

    assert(1 == len(set(map(lambda d: out[d][tk], out))))
    assert(all(map(lambda d: d[-1] == out[d[0]][ck], [(octo, 0), (sep, 1), (nov, 2)])))
    assert(all(map(lambda d: d[-1] == out[d[0]][ik], [(octo, 2), (sep, 1), (nov, 0)])))
    assert(all(map(lambda d: d[-1] == out[d[0]][ak], [(octo, 0.0), (sep, 0.5), (nov, 1.0)])))

def test_filter_by_total():
    sep, octo, nov, t = '09/01/2005', '10/01/2005', '11/01/2005', 'total'
    in_data = {sep: {t: 4}, octo: {t: 3}, nov: {t: 7}}

    out_all = model.filter_by_total(acc_by_date=in_data, lowest_val=2)
    out_two = model.filter_by_total(acc_by_date=in_data, lowest_val=3)
    out_empty = model.filter_by_total(acc_by_date=in_data, lowest_val=400)

    assert({sep, octo, nov} == set(out_all.keys()))
    assert({sep, nov} == set(out_two))
    assert({} == {} if out_empty.keys() == [] else None)
    
