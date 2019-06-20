import tensorflow as tf
import logging
import csv
import numpy as np

#take this in as an argument
DATA_PATH = "/home/tanderson/datasets/cfb/cfbstats-com-2005-1-5-0/team-game-statistics.csv"

def read_file(**kwargs):
    input_file = kwargs['input_file']
    result = {}

    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            game_code_key, team_code_key = "Game Code", "Team Code"
            game_code_id, team_code_id = row[game_code_key], row[team_code_key]
            team_game_stats = {k: row[k] for k in set(row.keys()).difference({game_code_key, team_code_key})}
            if game_code_id in result:
                result[game_code_id][team_code_id] = team_game_stats
            else:
                result[game_code_id] = {}
                result[game_code_id][team_code_id] = team_game_stats
    return result

def check_team_stats():
    correct, incorrect = [], []
    for k, v in team_stats.iteritems():
        if len(v.keys()) == 2:
            correct.append(k)
        else:
            incorrect.append(k)
    
    print("Correct len: %d, incorrect len: %d, total: %d", len(correct), len(incorrect), 
          len(correct) + len(incorrect))    

def main(unused_argv):
    team_stats = read_file(input_file=DATA_PATH)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
