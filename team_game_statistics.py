import utilities as util
import operator as op
import copy
import os.path as path

#functions for reading in data from team_game_statistics.csv

FILE_NAME = 'team-game-statistics.csv'

def csv_to_map(csv_reader):
    """Given a list of values, converts each value to a map

    Args:
         csv_reader: a list of values
    
    Returns: map of game ids to a map of team ids and their game stats
    """

    result = {}
    for row in csv_reader:
        game_code_key, team_code_key = 'Game Code', 'Team Code'
        game_code_id, team_code_id = row[game_code_key], row[team_code_key]
        team_game_stats = util.subset_of_map(full_map=row, 
                                             take_out_keys={game_code_key, team_code_key})
        if game_code_id not in result:
            result[game_code_id] = {}
        result[game_code_id][team_code_id] = team_game_stats
    return result

def alter_types(game_map, type_mapper):
    """Applies a type_mapper function over the values of a game_map

    Args:
         game_map: map of game ids to their stat values
         type_mapper: function to map each team's stats for a given game
    
    Returns: a map of game ids to their values
    """

    result = {}
    for game_id, teams in game_map.items():
        result[game_id] = {}
        for team_id, team_stats in teams.items():
            result[game_id][team_id] = type_mapper(team_stats=team_stats)
    return result

def type_mapper(team_stats):
    """Casts the values of a given map to either int or float. The 
       first type that produces a successful cast is the resulting
       type of the value

    Args:
         team_stats: map of stats to values for a given game
    
    Returns: a map
    """

    result = {}
    for stat_name, stat_value in team_stats.items():
        value = util.convert_type(types=(int, float), value=stat_value)
        result[stat_name] = value
    return result

def add_labels(team_game_stats):
    """Adds a winning key to each game in team_game_stats, where the 
       value has the winning team, total points, and points per team

    Args:
         team_game_stats: map of game ids to the statistics by each 
                          team for that game
    
    Returns: a map
    """

    team_game_stats_copy = copy.deepcopy(team_game_stats)
    for game_id, teams in team_game_stats_copy.items():
        points = []
        for team_id, game_stats in teams.items():
            points.append((game_stats['Points'], team_id))
        team_game_stats_copy[game_id]['Winner'] = {'Team Code': max(points)[1], 
                                              'Total Points': sum([t[0] for t in points]), 
                                              points[0][1]: points[0][0], 
                                              points[-1][1]: points[-1][0]}
    return team_game_stats_copy

def win_loss_pct(tid1, games):
    """Computes winning percentage for a given team across a
       given number of games

    Args:
         tid1: team id
         games: games in which tid1 is a participant
    
    Returns: winning percentage
    """

    ps = 'Points'
    result = {}
    result[tid1] = 0.0
    for gid, stat in games.items():
        win_tid = max([(stat[t][ps], t) for t in stat])[1]
        if win_tid == tid1:
            result[tid1] += 1
    result[tid1] = result[tid1] / len(games) if len(games) > 0 else 0.0

    return result

def averages(game_stats, team_ids):
    """Computes averages of all statistics in the game_stats map for a 
       given set of team_ids

    Args:
         game_stats: map of game ids to their statistics by team id
         team_ids: set of team_ids to compute averages
    
    Returns: map of averages by team id
    """

    avgs = {}
    for gid, team_stats in game_stats.items():
        for tid, stats in team_stats.items():
            if tid in team_ids:
                if tid in avgs:
                    avgs[tid] = util.merge_maps(map1=avgs[tid], map2=stats, merge_op=op.add)
                else:
                    avgs[tid] = stats
    for tid, stats in avgs.items():
        avgs[tid] = util.merge_maps(map1=stats, 
                                    map2=util.create_map(keys=list(avgs[tid].keys()), 
                                                         default=float(len(list(game_stats.keys())))), 
                                    merge_op=op.truediv)
    return avgs

def team_game_stats(directory):
    """Reads in data from a team-game-statistics.csv file

    Args:
         directory: directory containing a team-game-statistics.csv 
                    file
    
    Returns: a map of data of each game by teams
    """

    tgs_file = path.join(directory, FILE_NAME)
    team_game_stats = util.read_file(input_file=tgs_file, func=csv_to_map)
    converted_tgs = alter_types(type_mapper=type_mapper, 
                                    game_map=team_game_stats)
    
    return converted_tgs

def tgs_data(dirs):
    return {d: team_game_stats(directory=d) for d in dirs}

def add_labels_season(team_game_stats):
    return {k: add_labels(team_game_stats=team_game_stats[k]) for k in team_game_stats}


