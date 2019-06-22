import utilities as util
import operator as op

#functions for reading in data from team_game_statistics.csv

FILE_NAME = 'team-game-statistics.csv'

def csv_to_map(**kwargs):
    csv_reader = kwargs['csv_reader']
    result = {}

    for row in csv_reader:
        game_code_key, team_code_key = 'Game Code', 'Team Code'
        game_code_id, team_code_id = row[game_code_key], row[team_code_key]
        team_game_stats = util.subset_of_map(full_map=row, 
                                             take_out_keys={game_code_key, team_code_key})
        if game_code_id in result:
            result[game_code_id][team_code_id] = team_game_stats
        else:
            result[game_code_id] = {}
            result[game_code_id][team_code_id] = team_game_stats
    return result

def alter_types(**kwargs):
    game_map, type_mapper = kwargs['game_map'], kwargs['type_mapper']
    result = {}

    for game_id, teams in game_map.iteritems():
        result[game_id] = {}
        for team_id, team_stats in teams.iteritems():
            result[game_id][team_id] = type_mapper(team_stats=team_stats)
    return result

def type_mapper(**kwargs):
    team_stats = kwargs['team_stats']
    result = {}

    for stat_name, stat_value in team_stats.iteritems():
        value = util.convert_type(types=(int, float), value=stat_value)
        result[stat_name] = value
    return result

def add_labels(**kwargs):
    team_game_stats = kwargs['team_game_stats']
    
    for game_id, teams in team_game_stats.iteritems():  
        points = []
        for team_id, game_stats in teams.iteritems():
            points.append((game_stats['Points'], team_id))
        team_game_stats[game_id]['Winner'] = {'Team Code': max(points)[1], 
                                              'Total Points': sum(map(lambda t: t[0], points))}
    return team_game_stats

def averages(**kwargs):
    game_stats, team_ids = kwargs['game_stats'], kwargs['team_ids']

    avgs = {}
    for gid, team_stats in game_stats.iteritems():
        for tid, stats in team_stats.iteritems():
            if tid in team_ids:
                if tid in avgs:
                    avgs[tid] = util.merge_maps(map1=avgs[tid], map2=stats, merge_op=op.add)
                else:
                    avgs[tid] = stats
    for tid, stats in avgs.iteritems():
        avgs[tid] = util.merge_maps(map1=stats, 
                                    map2=util.create_map(keys=avgs[tid].keys(), default=float(len(game_stats.keys()))), 
                                    merge_op=op.div)
    return avgs
