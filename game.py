import utilities as util

#functions for reading in data from game.csv

FILE_NAME = 'game.csv'

def csv_to_map(**kwargs):
    csv_reader = kwargs['csv_reader']
    result = {}

    for row in csv_reader:
        game_code_id = "Game Code"
        result[row[game_code_id]] = {k: row[k] for k in set(row.keys())
                           .difference({game_code_id})}
    return result
