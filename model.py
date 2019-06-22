import tensorflow as tf
import logging
import numpy as np
import utilities as util
import team_game_statistics as tgs
import os.path as path
import game as game
import operator as op

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        print("v " + str(v))
        print("len(v) " + str(len(v)))
        raw_input() 

def team_game_stats(**kwargs):
    input_directory = kwargs['directory']

    tgs_file = path.join(input_directory, tgs.FILE_NAME)
    team_game_stats = util.read_file(input_file=tgs_file, func=tgs.csv_to_map)
    converted_tgs = tgs.alter_types(type_mapper=tgs.type_mapper, 
                                    game_map=team_game_stats)
    #labels = tgs.add_labels(team_game_stats=converted_tgs)
    
    return converted_tgs

def game_stats(**kwargs):
    input_directory = kwargs['directory']

    game_file = path.join(input_directory, game.FILE_NAME)
    game_data = util.read_file(input_file=game_file, func=game.csv_to_map)
    
    return game_data

def main(args):
    if len(args) == 2:
        input_directory = args[1]
        stats = team_game_stats(directory=input_directory)
        game_data = game_stats(directory=input_directory)
        games_by_team = game.seasons_by_game_code(games=game_data, 
                                                     game_code_id='0365002820050910')
        gb = game.subseason(team_games=games_by_team['365'], game_code_id='0031036520051125', 
                               compare=op.le)
        games_to_avg = {gid: stats[gid] for gid in map(lambda g: g[0], gb)}
        avgs = tgs.averages(game_stats=games_to_avg, team_ids={'365'})

        loop_through(data=avgs)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
