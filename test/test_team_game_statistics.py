import team_game_statistics as tgs
import copy

def test_csv_to_map():
    gc, tc, sk, gid_1, gid_2, ts = 'Game Code', 'Team Code', 'score', '1', '2', 50
    in_data = [{gc: '1', sk: ts, tc: '1'}, {gc: '1', sk: ts, tc: '2'}, 
               {gc: '2', sk: ts, tc: '3'}, {gc: '2', sk: ts, tc: '4'}]
    
    out = tgs.csv_to_map(csv_reader=in_data)

    assert(len(out) == 2)
    assert(set(map(lambda e: e[gc], in_data)) == set(out.keys()))
    assert(set(map(lambda e: e[tc], in_data)) == set(reduce(lambda l1,l2: l1 + l2,
        map(lambda gid: out[gid].keys(), out))))
    for gid, teams in out.iteritems():
        for tid, stat in teams.iteritems():
            assert({sk} == set(stat.keys()))
            assert({ts} == set(stat.values()))

def test_alter_types():
    gid, tid_1, tid_2, s, v1, v2 = '1', '10', '11', 'Points', '23', '0.23'
    in_data = {gid: {tid_1: {s: v1}, tid_2: {s: v2}}}

    out = tgs.alter_types(game_map=in_data, type_mapper=tgs.type_mapper)

    assert(int(v1) == out[gid][tid_1][s])
    assert(float(v2) == out[gid][tid_2][s])

def test_add_labels():
    gid, tid_1, tid_2, s, v1, v2, w, tc, tp = '1', '10', '11', 'Points', 23,\
                                          46, 'Winner', 'Team Code', 'Total Points'
    tp_val = v1 + v2
    in_data = {gid: {tid_1: {s: v1}, tid_2: {s: v2}}}

    out = tgs.add_labels(team_game_stats=in_data)
    out_win = out[gid][w]

    assert(out_win[tc] == tid_2)
    assert(out_win[tp] == tp_val)
    assert(out_win[tid_1] == v1)
    assert(out_win[tid_2] == v2)

def test_win_loss_pct():
    #not used by model at this point, skipping test for it
    pass

def test_averages():
    ps, tid_1, tid_2, tid_3, t1_val, t2_val = 'Points', '7', '8', '42', 4, 8
    #have 3 tids for a game to test inclusivity of team_ids
    in_data = {str(gid): {tid_1: {ps: t1_val}, tid_2: {ps: t2_val}, tid_3: {ps: 1}} for gid in range(3)}

    out = tgs.averages(game_stats=in_data, team_ids={tid_1, tid_2})

    assert({tid_1, tid_2} == set(out.keys()))
    assert(out[tid_1][ps] == t1_val)
    assert(out[tid_2][ps] == t2_val)
    
