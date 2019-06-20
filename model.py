import tensorflow as tf
import logging
import csv
import numpy as np

#checking functions, these are defensive functions that make sure the values read in are correct
def check_team_stats(**kwargs):
    team_stats = kwargs['team_stats']

    correct, incorrect = [], []
    for k, v in team_stats.iteritems():
        if len(v.keys()) == 2:
            correct.append(k)
        else:
            incorrect.append(k)
    
    print("Correct len: %d, incorrect len: %d, total: %d", len(correct), len(incorrect), 
          len(correct) + len(incorrect)) 

def team_game_statistics(**kwargs):
    csv_reader = kwargs['csv_reader']
    result = {}

    for row in csv_reader:
        game_code_key, team_code_key = "Game Code", "Team Code"
        game_code_id, team_code_id = row[game_code_key], row[team_code_key]
        team_game_stats = {k: row[k] for k in set(row.keys())
                           .difference({game_code_key, team_code_key})}
        if game_code_id in result:
            result[game_code_id][team_code_id] = team_game_stats
        else:
            result[game_code_id] = {}
            result[game_code_id][team_code_id] = team_game_stats
    return result    

def read_file(**kwargs):
    input_file, func = kwargs['input_file'], kwargs['func']
    result = None

    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        result = func(csv_reader=csv_reader)
    return result

#take this in as an argument
DATA_PATH = "/home/tanderson/datasets/cfb/cfbstats-com-2005-1-5-0/team-game-statistics.csv"

def main(unused_argv):
    team_stats = read_file(input_file=DATA_PATH, func=team_game_statistics)
    check_team_stats(team_stats=team_stats)

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
