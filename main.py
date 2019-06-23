import tensorflow as tf
import logging
import numpy as np
import utilities as util
import team_game_statistics as tgs
import os.path as path
import game as game
import operator as op
import model

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        #print("v " + str(v))
        for attr, av in v.iteritems():
            print("attr: %s av: %s" % (str(attr), str(av)))
        print("len(v) " + str(len(v)))
        raw_input() 

def team_game_stats(**kwargs):
    input_directory = kwargs['directory']

    tgs_file = path.join(input_directory, tgs.FILE_NAME)
    team_game_stats = util.read_file(input_file=tgs_file, func=tgs.csv_to_map)
    converted_tgs = tgs.alter_types(type_mapper=tgs.type_mapper, 
                                    game_map=team_game_stats)

    #this breaks statistics
    #labels = tgs.add_labels(team_game_stats=converted_tgs)
    
    return converted_tgs

def game_stats(**kwargs):
    input_directory = kwargs['directory']

    game_file = path.join(input_directory, game.FILE_NAME)
    game_data = util.read_file(input_file=game_file, func=game.csv_to_map)
    
    return game_data

def team_avgs_by_gid(**kwargs):
    game_code_id = kwargs['game_code_id']

    games_by_team = game.seasons_by_game_code(games=game_data, 
                                              game_code_id=game_code_id)
    avgs = []
    for tid, games in games_by_team.iteritems():
        gb = game.subseason(team_games=games_by_team[tid], game_code_id=game_code_id, 
                           compare=op.le)  
        games_to_avg = {gid: stats[gid] for gid in map(lambda g: g[0], gb)}
        avgs.append((tid, tgs.averages(game_stats=games_to_avg, team_ids={tid})))
    
    return avgs
    

def main(args):
    if len(args) == 2:
        input_directory = args[1]
        stats = team_game_stats(directory=input_directory)
        game_data = game_stats(directory=input_directory)

        avgs = team_avgs_by_gid(game_code_id='0365002820050910')

        print(model.eval_func(stat_map1=avgs['365'], 
                              stat_map2=avgs['365'], st1_key='365_1', st2_key='365_2', 
                              field_win=model.FIELD_WIN_SEMANTICS, 
                              undec_fields=model.UNDECIDED_FIELDS))
        #loop_through(data=avgs)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
