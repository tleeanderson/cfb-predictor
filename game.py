import utilities as util
from dateutil import parser as du
import operator as op

#functions for reading in data from game.csv

FILE_NAME = 'game.csv'

def csv_to_map(**kwargs):
    csv_reader = kwargs['csv_reader']
    result = {}

    for row in csv_reader:
        game_code_id = 'Game Code'
        result[row[game_code_id]] = {k: row[k] for k in set(row.keys())
                           .difference({game_code_id})}
    return result

def seasons_by_game_code(**kwargs):
    game_code_id, games = kwargs['game_code_id'], kwargs['games']
    visit_team, home_team = 'Visit Team Code', 'Home Team Code'

    games_by_team = {}
    games_by_team[games[game_code_id][visit_team]] = {}
    games_by_team[games[game_code_id][home_team]] = {}

    for gid, game_info in games.iteritems():
        if game_info[visit_team] in games_by_team:
            games_by_team[game_info[visit_team]][gid] = util.subset_of_map(full_map=game_info, 
                                                                           take_out_keys={'Game Code'})
        if game_info[home_team] in games_by_team:
            games_by_team[game_info[home_team]][gid] = util.subset_of_map(full_map=game_info, 
                                                                           take_out_keys={'Game Code'})
    return games_by_team

def subseason(**kwargs):
    team_games, game_code_id, compare, date = kwargs['team_games'], kwargs['game_code_id'],\
                                              kwargs['compare'], 'Date'
    
    if game_code_id in team_games:
        lis = map(lambda e: (e[0], du.parse(e[-1][date])), team_games.iteritems())
        lis.sort(key=lambda e: e[-1])
        return filter(lambda x: compare(x[-1], du.parse(team_games[game_code_id][date])), lis)
    else:
        raise ValueError("%s id was not in %s" % (str(game_code_id), str(team_games)))
    
