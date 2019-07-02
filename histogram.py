import matplotlib.pyplot as plt
import main as temp_lib
import estimator as est
import model
import team_game_statistics as tgs
import sys
import glob
import os.path as path

def reorder(**kwargs):
    fs = kwargs['features']

    res = []
    for field in filter(lambda x: '-0' in x, set(fs.keys())):
        lis = filter(lambda e: e[0 : len(e) - 2] == field[0 : len(field) - 2], 
                     filter(lambda x: '-1' in x, set(fs.keys())))
        if len(lis) == 1:
            res.append((field, fs[field]))
            res.append((lis[0], fs[lis[0]]))
        else:
            print("Could not find match for field %s, lis %s" % (str(field), str(lis)))

    return res

def histogram(**kwargs):
    avgs, team_stats = kwargs['avgs'], kwargs['team_stats']

    fields, _ = est.input_data(game_averages=avgs, labels=tgs.add_labels(team_game_stats=team_stats))

    data = reorder(features=fields)

    for i in range(1, len(data), 2):
        plt.figure(num=i)
        plt.hist(data[i - 1][1], bins='auto')
        plt.title(data[i - 1][0])

        plt.hist(data[i][1], bins='auto')
        plt.title(data[i][0])

    plt.show()

def avgs_by_file(**kwargs):
    d, pre = kwargs['directory'], kwargs['prefix']
    
    avgs_by_file = {}
    for data_dir in glob.glob(path.join(d, pre)):
        stats = temp_lib.team_game_stats(directory=data_dir)
        game_data = temp_lib.game_stats(directory=data_dir)
        
        avgs = est.averages(team_game_stats=stats, game_infos=game_data, skip_fields=model.UNDECIDED_FIELDS)
        avgs_by_file[data_dir] = avgs

        histogram(avgs=avgs, team_stats=stats)
    return avgs_by_file

def main(args):
    if len(args) == 3:
        avgs_by_file(directory=args[1], prefix=args[2])
    else:
        print("Usage: %s [directory] [prefix]" % ('./' + str(args[0])))

if __name__ == '__main__':
    main(sys.argv)
