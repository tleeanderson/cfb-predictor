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
