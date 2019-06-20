import tensorflow as tf
import logging
import numpy as np
import utilities as util
import team_game_statistics as tgs

def loop_through(**kwargs):
    data = kwargs['data']

    for k, v in data.iteritems():
        print("k " + str(k))
        print("v " + str(v['Winner']))
        raw_input()        

def main(args):
    if len(args) == 2:
        input_file = args[1]
        team_game_stats = util.read_file(input_file=input_file, func=tgs.csv_to_map)
        converted_tgs = tgs.alter_types(type_mapper=tgs.type_mapper, 
                                        game_map=team_game_stats)
        labels = tgs.add_labels(team_game_stats=converted_tgs)
        loop_through(data=labels)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
