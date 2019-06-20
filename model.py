import tensorflow as tf
import logging
import numpy as np
import utilities as util
import team_game_statistics as tgs

def game_code_to_winning_team(**kwargs):
    team_game_stats = kwargs['team_game_stats']
    result = {}

    for k, v in team_game_stats.iteritems():
        print("Type of k: " + str(type(k)))
        print("Type of v: " + str(v))
        raw_input()        

def main(args):
    if len(args) == 2:
        input_file = args[1]
        team_game_stats = util.read_file(input_file=input_file, func=tgs.csv_to_map)
        converted_tgs = tgs.alter_types(type_mapper=tgs.type_mapper, 
                                        game_map=team_game_stats)

        game_code_to_winning_team(team_game_stats=converted_tgs)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
