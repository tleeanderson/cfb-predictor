import operator as op
import game
import team_game_statistics as tgs

MAX_COMPARE = lambda v1, v2: attr_compare(v1=v1, v2=v2, compare=max)
MIN_COMPARE = lambda v1, v2: attr_compare(v1=v1, v2=v2, compare=min)
UNDECIDED_FIELDS = {'Field Goal Att', 'Rush Att', 'Time Of Possession', 'Kickoff', '1st Down Pass', 
'Red Zone Field Goal', 'Kickoff Onside', 'Def 2XP Att', 'Off XP Kick Made', '1st Down Rush', 'Def 2XP Made', 
'Pass TD', 'Field Goal Made', 'Pass Comp', 'Rush TD', 'Misc Ret', 'Kickoff Touchback', 'Off 2XP Att',
'Fourth Down Att', 'Off XP Kick Att', 'Pass Att', 'Off 2XP Made', 'Kickoff Ret', 'Pass Conv', 'Win Loss Percentage'}

ORIGINAL_MAX = ('Red Zone Att', 'Tackle For Loss Yard', 'Fourth Down Conv', 'QB Hurry', 'Safety',
'Int Ret', 'Int Ret TD', 'Punt Ret Yard', 'Kickoff Yard', 'Fum Ret Yard', 'Pass Yard', 'Fumble Forced', 
'Red Zone TD', 'Misc Ret TD', 'Tackle Assist', 'Kickoff Ret TD', 'Punt Ret', 'Fum Ret', 'Points', 
'Misc Ret Yard', 'Pass Int', 'Int Ret Yard', 'Kick/Punt Blocked', 'Punt Yard', 
'Tackle For Loss', 'Kickoff Ret Yard', 'Third Down Conv', 'Fum Ret TD', 'Tackle Solo', 'Sack', 'Rush Yard',
'Sack Yard', 'Punt Ret TD', 'Pass Broken Up')
ORIGINAL_MIN = ('Kickoff Out-Of-Bounds', 'Penalty', 'Fumble', 'Penalty Yard', 'Punt', '1st Down Penalty', 
'Third Down Att', 'Fumble Lost')

CURRENT_MIN = ('Punt', '1st Down Penalty', 'Pass Int', 'Penalty', 'Fumble', 'Third Down Att', 'Fumble Lost', 
               'Penalty Yard')
CURRENT_MAX = ('Red Zone Att', 'Points', 'Pass Yard', 'Fumble Forced', 'Tackle For Loss', 
'Tackle For Loss Yard', 'Kickoff Ret Yard', 'Red Zone TD', 'Fourth Down Conv', 'Pass Broken Up', 'Punt Ret', 
'Kickoff Yard', 'Tackle Solo', 'Rush Yard', 'Punt Ret Yard', 'Tackle Assist', 'Sack', 'Sack Yard',
'Int Ret Yard', 'Int Ret', 'Punt Yard', 'Third Down Conv')

LEFTOVER_MAX = set(['Fum Ret', 'Pass Int', 'Fum Ret TD', 'Kickoff Ret TD', 'Fum Ret Yard', 'QB Hurry', 
'Misc Ret TD', 'Misc Ret Yard', 'Int Ret TD', 'Safety', 'Kick/Punt Blocked', 'Punt Ret TD'])
LEFTOVER_MIN = set(['Fumble Lost', 'Penalty Yard', 'Kickoff Out-Of-Bounds'])

FIELD_WIN_SEMANTICS = {CURRENT_MAX: MAX_COMPARE, CURRENT_MIN: MIN_COMPARE}

def attr_compare(**kwargs):
    v1, v2, compare = kwargs['v1'], kwargs['v2'], kwargs['compare']
    if v1 == v2:
        return 'tie'
    else:
        return ([v1] + [v2]).index(compare(v1, v2))

def evaluation(**kwargs):
    stat_map1, st1_key, stat_map2, st2_key, field_win, undec_fields, c1, c2 = kwargs['stat_map1'],\
                                                                             kwargs['st1_key'],\
                                                                             kwargs['stat_map2'],\
                                                                             kwargs['st2_key'],\
                                                                             kwargs['field_win'],\
                                                                             kwargs['undec_fields'], 0, 0
    
    for sm1_key in stat_map1.keys():
        if sm1_key in undec_fields:
            continue
        else:
            lis = filter(lambda t: t[0], map(lambda k: (sm1_key in k, k), field_win))
            if not lis:
                raise ValueError("key %s is not in %s" % (str(sm1_key), str(field_win)))
            func_key = lis[0][1]
            ind = field_win[func_key](stat_map1[sm1_key], stat_map2[sm1_key])
            if ind == 'tie':
                #tie does not affect scores for either team
                pass
            elif ind == 0:
                c1 += 1
            else:
                c2 += 1

    return {st1_key: c1, st2_key: c2, 'tie': c1 == c2}

def predict(**kwargs):
    team_avgs, game_code_id, tg_stats, eval_func = kwargs['team_avgs'], kwargs['game_code_id'],\
                                                   kwargs['tg_stats'], kwargs['eval_func']

    keys = team_avgs.keys()
    if len(keys) == 2:
        prediction = eval_func(stat_map1=team_avgs[keys[0]], stat_map2=team_avgs[keys[1]], 
                               st1_key=keys[0], st2_key=keys[1], field_win=FIELD_WIN_SEMANTICS, 
                               undec_fields=UNDECIDED_FIELDS.union(LEFTOVER_MAX).union(LEFTOVER_MIN))
        prediction.update({'Winner': max(prediction.iteritems(), key=op.itemgetter(1))[0], 
                           'tie': prediction['tie']})
        
        return prediction
    else:
        raise ValueError("len(team_avgs.keys()) == 2 must be true, team_avgs: %s" % (str(team_avgs)))
    

def team_avgs(**kwargs):
    game_code_id, game_data, tg_stats = kwargs['game_code_id'], kwargs['game_data'], kwargs['tg_stats']

    games_by_team = game.seasons_by_game_code(games=game_data, 
                                              game_code_id=game_code_id)
    avgs = {}
    for tid, games in games_by_team.iteritems():        
        gb = game.subseason(team_games=games, game_code_id=game_code_id, 
                           compare=op.lt)
        games_to_avg = {gid: tg_stats[gid] for gid in map(lambda g: g[0], gb)}
        avgs.update(tgs.averages(game_stats=games_to_avg, team_ids={tid}))        
        if len(gb) > 0:
            avgs[tid]['Win Loss Percentage'] = tgs.win_loss_pct(tid1=tid, games=games_to_avg)        
    return avgs

def predict_all(**kwargs):
    team_game_stats, game_infos, no_pred, no_pred_key = kwargs['team_game_stats'], kwargs['game_infos'],\
                                                        set(), kwargs['no_pred_key']

    preds = {}
    for gid in team_game_stats.keys():
        ta = team_avgs(game_code_id=gid, game_data=game_infos, tg_stats=team_game_stats)
        if len(ta) == 2:
            preds[gid] = predict(team_avgs=ta, game_code_id=gid, tg_stats=team_game_stats, 
                                 eval_func=evaluation)
        else:            
            if no_pred_key in preds:
                preds[no_pred_key].append(gid)
            else:
                preds[no_pred_key] = []

    return preds

def accuracy(**kwargs):
    tg_stats, predictions, winner, team_code, corr_key, incorr_key, total_key, sk, ak =\
    kwargs['tg_stats'], kwargs['predictions'], 'Winner', 'Team Code', kwargs['correct_key'],\
    kwargs['incorrect_key'], kwargs['total_key'], kwargs['skip_keys'], kwargs['acc_key']

    result = {}
    for gid, pred in predictions.iteritems():
        if gid in sk:
            continue
        actual = tg_stats[gid][winner][team_code]
        p = pred[winner]
        if actual == p and not pred['tie']:
            if corr_key in result:
                result[corr_key] += 1
            else:
                result[corr_key] = 1
        else:
            if incorr_key in result:
                result[incorr_key] += 1
            else:
                result[incorr_key] = 1

    result[total_key] = result[corr_key] + result[incorr_key]
    result[ak] = float(result[corr_key]) / result[total_key]
    return result

def accuracy_by_date(**kwargs):
    tg_stats, preds, gi, winner, tc, dt, ck, ik, tk, ak, sk = kwargs['tg_stats'], kwargs['predictions'],\
                                                          kwargs['game_info'], 'Winner', 'Team Code',\
                                                          'Date', kwargs['correct_key'], kwargs['incorrect_key'],\
                                                          kwargs['total_key'], kwargs['acc_key'],\
                                                          kwargs['skip_keys']

    result = {}
    for gid, info in preds.iteritems():
        if gid in sk:
            continue
        info = gi[gid]
        actual = tg_stats[gid][winner][tc]
        p = preds[gid][winner]
        if info[dt] not in result:
            result[info[dt]] = {}
        def_vals = filter(lambda k: result[info[dt]].get(k.keys()[0]) is None, 
                          [{ck: 0}, {ak: 0}, {ik: 0}, {tk: 0}])
        for df in def_vals:
            result[info[dt]].update(df)
        if actual == p:
            result[info[dt]][ck] += 1
        else:
            result[info[dt]][ik] += 1

    for k in result.keys():
        result[k][tk] = result[k][ck] + result[k][ik]
        result[k][ak] = float(result[k][ck]) / result[k][tk] if result[k][tk] > 0 else 0.0

    return result

def filter_by_total(**kwargs):
    abd, hi = kwargs['acc_by_date'], kwargs['hi_total']

    return {d: abd[d] for d in filter(lambda d: abd[d]['total'] > hi, abd)}

def print_list():
        #     lis = list(accuracy_by_date.iteritems())
        # lis.sort(key=lambda x: du.parse(x[0]))
        # for date in lis:
        #     print(date)
        # raw_input()
    pass
        
