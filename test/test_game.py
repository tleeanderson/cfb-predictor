import game
import operator as op

def test_csv_to_map():
    gc, sk = 'Game Code', 'score'
    in_data = map(lambda i: {gc: str(i), sk: 50 + i}, range(10))
    
    out = game.csv_to_map(csv_reader=in_data)

    assert(len(out) == len(in_data))
    for gid, val in out.iteritems():
        assert gc not in val.keys() 
        assert sk in val.keys()

def test_seasons_by_game_code():
    visit_team, home_team = 'Visit Team Code', 'Home Team Code'
    v_tid, h_tid, gid = '12', '22', str(9)
    games = {gid: {visit_team: v_tid, home_team: h_tid}, 
             str(1): {visit_team: '-1', home_team: '-1'}, 
             str(2): {visit_team: h_tid, home_team: '-1'}, 
             str(3): {visit_team: '-1', home_team: v_tid}}

    out = game.seasons_by_game_code(game_code_id=gid, games=games)

    assert(v_tid in out)
    assert(h_tid in out)
    assert(len(out) == 2)
    assert(len(out[v_tid]) == 2)
    assert({str(3), gid} == set(out[v_tid].keys()))
    assert(len(out[h_tid]) == 2 )
    assert({str(2), gid} == set(out[h_tid].keys()))

def test_subseason():
    tid, gid_1, gid_2, gid_3, sep, octo, nov, d = '1', '10', '11', '12', '09/01/2005',\
                                               '10/01/2005', '11/01/2005', 'Date'
    games = {gid_1: {d: sep}, gid_2: {d: octo}, 
             gid_3: {d: nov}}
    out1 = game.subseason(team_games=games, game_code_id=gid_3, compare=op.lt)
    out2 = game.subseason(team_games=games, game_code_id=gid_2, compare=op.lt)
    out3 = game.subseason(team_games=games, game_code_id=gid_2, compare=op.gt)

    assert(len(out1) == 2)
    assert(out1[0][0] == gid_1)
    assert(out1[1][0] == gid_2)
    
    assert(len(out2) == 1)
    assert(out2[0][0] == gid_1)

    assert(len(out3) == 1)
    assert(out3[0][0] == gid_3)

    try:
        game.subseason(team_games=games, game_code_id='45', compare=op.lt)
    except ValueError:
        assert(True)
    else:
        assert(False)
