import tensorflow as tf
import logging
import numpy as np
import utilities as util
import team_game_statistics as tgs
import os.path as path
import game as game

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        print("v " + str(v))
        raw_input() 

def main(args):
    if len(args) == 2:
        input_directory = args[1]
        tgs_file = path.join(input_directory, tgs.FILE_NAME)
        game_file = path.join(input_directory, game.FILE_NAME)

        team_game_stats = util.read_file(input_file=tgs_file, func=tgs.csv_to_map)
        converted_tgs = tgs.alter_types(type_mapper=tgs.type_mapper, 
                                        game_map=team_game_stats)
        labels = tgs.add_labels(team_game_stats=converted_tgs)

        game_data = util.read_file(input_file=game_file, func=game.csv_to_map)
        loop_through(data=game_data)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
