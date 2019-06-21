import utilities as util

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

def team_games_by_game_code(**kwargs):
    game_code_id, games = kwargs['game_code_id'], kwargs['games']
    visit_team, home_team = 'Visit Team Code', 'Home Team Code'

    games_by_team = {}
    games_by_team[games[game_code_id][visit_team]] = []
    games_by_team[games[game_code_id][home_team]] = []

    for gid, game_info in games.iteritems():
        if game_info[visit_team] in games_by_team:
            games_by_team[game_info[visit_team]].append({gid: game_info})
        if game_info[home_team] in games_by_team:
            games_by_team[game_info[home_team]].append({gid: game_info})

    return games_by_team
        
        
            
        

    
