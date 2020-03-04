import operator as op
import game
import team_game_statistics as tgs

MAX_COMPARE = lambda v1, v2: attr_compare(v1=v1, v2=v2, compare=max)
MIN_COMPARE = lambda v1, v2: attr_compare(v1=v1, v2=v2, compare=min)
# UNDECIDED_FIELDS = {'Field Goal Att', 'Rush Att', 'Time Of Possession', 'Kickoff', '1st Down Pass', 
# 'Red Zone Field Goal', 'Kickoff Onside', 'Def 2XP Att', 'Off XP Kick Made', '1st Down Rush', 'Def 2XP Made', 
# 'Pass TD', 'Field Goal Made', 'Pass Comp', 'Rush TD', 'Misc Ret', 'Kickoff Touchback', 'Off 2XP Att',
# 'Fourth Down Att', 'Off XP Kick Att', 'Pass Att', 'Off 2XP Made', 'Kickoff Ret', 'Pass Conv', 'Win Loss Percentage'}

UNDECIDED_FIELDS = {'Field Goal Att', 'Rush Att', 'Time Of Possession', 'Kickoff', '1st Down Pass', 
'Red Zone Field Goal', 'Kickoff Onside', 'Def 2XP Att', 'Off XP Kick Made', '1st Down Rush', 'Def 2XP Made', 
'Pass TD', 'Field Goal Made', 'Pass Comp', 'Rush TD', 'Misc Ret', 'Kickoff Touchback', 'Off 2XP Att',
'Fourth Down Att', 'Off XP Kick Att', 'Pass Att', 'Off 2XP Made', 'Kickoff Ret', 'Pass Conv'}

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
# CURRENT_MAX = ('Red Zone Att', 'Points', 'Pass Yard', 'Fumble Forced', 'Tackle For Loss', 
# 'Tackle For Loss Yard', 'Kickoff Ret Yard', 'Red Zone TD', 'Fourth Down Conv', 'Pass Broken Up', 'Punt Ret', 
# 'Kickoff Yard', 'Tackle Solo', 'Rush Yard', 'Punt Ret Yard', 'Tackle Assist', 'Sack', 'Sack Yard',
# 'Int Ret Yard', 'Int Ret', 'Punt Yard', 'Third Down Conv')

CURRENT_MAX = ('Red Zone Att', 'Points', 'Pass Yard', 'Fumble Forced', 'Tackle For Loss', 
'Tackle For Loss Yard', 'Kickoff Ret Yard', 'Red Zone TD', 'Fourth Down Conv', 'Pass Broken Up', 'Punt Ret', 
'Kickoff Yard', 'Tackle Solo', 'Rush Yard', 'Punt Ret Yard', 'Tackle Assist', 'Sack', 'Sack Yard',
'Int Ret Yard', 'Int Ret', 'Punt Yard', 'Third Down Conv')

LEFTOVER_MAX = set(['Fum Ret', 'Pass Int', 'Fum Ret TD', 'Kickoff Ret TD', 'Fum Ret Yard', 'QB Hurry', 
'Misc Ret TD', 'Misc Ret Yard', 'Int Ret TD', 'Safety', 'Kick/Punt Blocked', 'Punt Ret TD', 
                    'Win Loss Percentage'])
LEFTOVER_MIN = set(['Fumble Lost', 'Penalty Yard', 'Kickoff Out-Of-Bounds'])

FIELD_WIN_SEMANTICS = {CURRENT_MAX: MAX_COMPARE, CURRENT_MIN: MIN_COMPARE}

def attr_compare(v1, v2, compare):
    """Compares two values and returns the index of the value in the 
       argument list.

    Args:
         v1: a value
         v2: a value
         compare: a function to compare v1 and v2
    
    Returns: index
    """

    if v1 == v2:
        return 'tie'
    else:
        return ([v1] + [v2]).index(compare(v1, v2))

def evaluation(stat_map1, st1_key, stat_map2, st2_key, field_win, undec_fields):
    """Evalutes two maps of statistics by comparing the alike fields according
       to field_win. The field that wins adds one to the total score of its
       associated map. A tie does not affect the score for either map.

    Args:
         stat_map1: map of stats
         st1_key: key to use in the output of stat_map1's score
         stat_map2: map of stats
         st2_key: key to use in the output of stat_map2's score
         field_win: map of tuples of fields to comparison function
         undec_fields: fields to be skipped
    
    Returns: map of scores
    """

    c1, c2 = 0, 0
    for sm1_key in list(stat_map1.keys()):
        if sm1_key in undec_fields:
            continue
        else:
            lis = [t for t in [(sm1_key in k, k) for k in field_win] if t[0]]
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

def predict(team_avgs, eval_func):
    """Predicts the winner between two maps of fields to their averages

    Args:
         team_avgs: the averaged data points for two teams
         eval_func: evaluation function to compare the team_avgs
                    maps
    
    Returns: prediction map
    """

    keys = list(team_avgs.keys())
    if len(keys) == 2:
        
        prediction = eval_func(stat_map1=team_avgs[keys[0]], stat_map2=team_avgs[keys[1]], 
                               st1_key=keys[0], st2_key=keys[1], field_win=FIELD_WIN_SEMANTICS, 
                               undec_fields=UNDECIDED_FIELDS.union(LEFTOVER_MAX).union(LEFTOVER_MIN))
        prediction.update({'Winner': max(iter(prediction.items()), key=op.itemgetter(1))[0], 
                           'tie': prediction['tie']})
        
        return prediction
    else:
        raise ValueError("len(team_avgs.keys()) == 2 must be true, team_avgs: %s" % (str(team_avgs)))
    

def team_avgs(game_code_id, game_data, tg_stats):
    """Computes the averages for all fields for two teams in a given game

    Args:
         game_code_id: game which serves as the basis for 
                       the computation of averages
         game_data: home, away etc. information for a game, see game.csv
         tg_stats: stats of all games by teams, see team-game-statistics.csv
    
    Returns: map of averaged data points for all fields for two teams
    """

    games_by_team = game.seasons_by_game_code(games=game_data, 
                                              game_code_id=game_code_id)
    avgs = {}
    for tid, games in games_by_team.items():        
        gb = game.subseason(team_games=games, game_code_id=game_code_id, 
                           compare=op.lt)
        games_to_avg = {gid: tg_stats[gid] for gid in [g[0] for g in gb]}
        avgs.update(tgs.averages(game_stats=games_to_avg, team_ids={tid}))

    return avgs

def predict_all(team_game_stats, game_infos, no_pred_key):
    """Runs predictions across all games in team_game_stats. See predict().

    Args:
         team_game_stats: stats of all games by teams, see team-game-statistics.csv
         game_infos: home, away etc. information for a game, see game.csv
         no_pred_key: key to use to associate games that cannot be predicted
    
    Returns: predictions
    """

    no_pred = set()
    preds = {}
    for gid in list(team_game_stats.keys()):
        ta = team_avgs(game_code_id=gid, game_data=game_infos, tg_stats=team_game_stats)
        if len(ta) == 2:
            preds[gid] = predict(team_avgs=ta, eval_func=evaluation)
        else:
            if no_pred_key not in preds:
                preds[no_pred_key] = []
            preds[no_pred_key].append(gid)
    return preds

def accuracy(tg_stats, predictions, correct_key, incorrect_key,
             total_key, skip_keys, acc_key, game_info=None):
    """Computes accuracy of a given predictions map.

    Args:
         tg_stats: stats of all games by teams, see team-game-statistics.csv
         predictions: map of game ids to predicted outcome
         correct_key: key to use for correct results
         incorrect_key: key to use for incorrect results
         total_key: key to use for total results
         skip_keys: keys to skip in computing accuracy
         acc_key: key to use for accuracy results
         game_info: dummy arg to comply with generic caller function
    
    Returns: map of accuracy of predictions
    """

    winner, team_code = 'Winner', 'Team Code'
    result = {}
    for gid, pred in predictions.items():
        if gid in skip_keys:
            continue
        actual = tg_stats[gid][winner][team_code]
        p = pred[winner]
        if actual == p and not pred['tie']:
            if correct_key in result:
                result[correct_key] += 1
            else:
                result[correct_key] = 1
        else:
            if incorrect_key in result:
                result[incorrect_key] += 1
            else:
                result[incorrect_key] = 1

    for k in set({correct_key, incorrect_key}).difference(result.keys()):
        result.update({k: 0})
    result[total_key] = result[correct_key] + result[incorrect_key]
    result[acc_key] = float(result[correct_key]) / result[total_key]
    return result

def accuracy_by_date(tg_stats, predictions, game_info, correct_key,
                     incorrect_key, total_key, acc_key, skip_keys):
    """Computes accuracy of a given predictions map, but does so by date.
       So each date will have a value similar to the output of accuracy()

    Args:
         tg_stats: stats of all games by teams, see team-game-statistics.csv
         predictions: map of game ids to predicted outcome
         game_info: home, away etc. information for a game, see game.csv
         correct_key: key to use for correct results
         incorrect_key: key to use for incorrect results
         total_key: key to use for total results
         acc_key: key to use for accuracy results
         skip_keys: keys to skip in computing accuracy
    
    Returns: map of accuracy of predictions by date
    """

    winner, tc, dt = 'Winner', 'Team Code', 'Date'
    result = {}
    for gid, info in predictions.items():
        if gid in skip_keys:
            continue
        info = game_info[gid]
        actual = tg_stats[gid][winner][tc]
        p = predictions[gid][winner]
        if info[dt] not in result:
            result[info[dt]] = {}
        default_values = [k for k in [{correct_key: 0}, {acc_key: 0}, {incorrect_key: 0}, {total_key: 0}] if result[info[dt]].get(list(k.keys())[0]) is None]
        for df in default_values:
            result[info[dt]].update(df)
        if actual == p:
            result[info[dt]][correct_key] += 1
        else:
            result[info[dt]][incorrect_key] += 1

    for k in list(result.keys()):
        result[k][total_key] = result[k][correct_key] + result[k][incorrect_key]
        result[k][acc_key] = float(result[k][correct_key]) / result[k][total_key] if result[k][total_key] > 0 else 0.0

    return result

def filter_by_total(acc_by_date, lowest_val):
    """Filters map by hi_total

    Args:
         acc_by_date: map of accuracy by date, see accuracy_by_date()
         hi_total: values must be higher than this
    
    Returns: map of date to their accuracy results
    """

    return {d: acc_by_date[d] for d in [d for d in acc_by_date if acc_by_date[d]['total'] > lowest_val]}

def averages(team_game_stats, game_info):
    result = {}
    for gid in team_game_stats.keys():
        teams = team_avgs(game_code_id=gid, game_data=game_info,
                                    tg_stats=team_game_stats)
        if len(teams) == 2:
            result[gid] = teams
    return result
        
def averages_season(tgs_data, game_data):
    return {k: averages(team_game_stats=tgs_data[k], game_info=game_data[k]) for k in tgs_data}

def predict_all_avgs(team_game_avgs):
    return {k: predict(team_avgs=team_game_avgs[k], eval_func=evaluation) for k in team_game_avgs}

def predict_all_season(season_tgs):
    return {k: predict_all_avgs(team_game_avgs=season_tgs[k]) for k in season_tgs}

def accuracy_season(labels, preds, skip_keys):
    return {k: accuracy(tg_stats=labels[k], predictions=preds[k], correct_key='correct', 
                               incorrect_key='incorrect', total_key='total', skip_keys=skip_keys, 
                               acc_key='accuracy') for k in labels}


