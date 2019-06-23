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
        print("v " + str(v))
        # for attr, av in v.iteritems():
        #     print("attr: %s av: %s" % (str(attr), str(av)))
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
    
def main(args):
    if len(args) == 2:
        input_directory = args[1]
        stats = team_game_stats(directory=input_directory)
        game_data = game_stats(directory=input_directory)

        avgs = team_avgs_by_gid(game_code_id='0365002820050910', game_data=game_data, tg_stats=stats)

        # print(model.evaluation(stat_map1=avgs[0][1][avgs[0][0]], 
        #                       stat_map2=avgs[1][1][avgs[1][0]], st1_key=avgs[0][0], st2_key=avgs[1][0], 
        #                       field_win=model.FIELD_WIN_SEMANTICS, 
        #                       undec_fields=model.UNDECIDED_FIELDS))
        
        #loop_through(data=avgs[0][1])

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
