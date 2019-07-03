import matplotlib.pyplot as plt
import main as temp_lib
import estimator as est
import model
import team_game_statistics as tgs
import sys
import glob
import os.path as path
from scipy.stats import shapiro
import numpy as np

#There seems to be a correlation between the statistic value of the Shapiro-Wilk test and the 
#"normality" of the given distribution. We will say that any distribution with a statistic 
#greater than or equal to .8 is a failure to reject H0. Also, the p values are incredibly 
#low for all tests, but I am unsure why at this point.
SW_NORM_THRESHOLD = 0.80

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

def shapiro_wilk(**kwargs):
    dist = kwargs['distributions']

    result = {}
    for k, d in dist.iteritems():
        result[k] = shapiro(d)
    
    return result

def print_shapiro_wilk(**kwargs):
    sw, keys = kwargs['shapiro_wilk'], kwargs['keys']

    for k in keys:
        print("Field: %s, statistic: %s, p-value: %s" % (str(k), str(sw[k][0]), str(sw[k][1])))

def similar_field(**kwargs):
    f, afs = kwargs['field'], kwargs['all_fields']

    res = filter(lambda e: e[0 : len(e) - 2] == f[0 : len(f) - 2], afs)
    return res[0] if res else None

def normality_filter(**kwargs):
    sw, th = kwargs['shapiro_wilk'], kwargs['threshold']
    
    result = {}
    for f, val in filter(lambda i: '-0' in i[0], set(sw.iteritems())):
        if val[0] >= th:
            result[f] = val
            sf = similar_field(field=f, all_fields=filter(lambda k: '-1' in k, set(sw.keys())))
            if sf:
                result[sf] = sw[sf]
    
    return result

def normal_dists(**kwargs):
    data = kwargs['data_by_file']
        
    fields, _ = est.input_data(game_averages=data['avgs'], 
                               labels=tgs.add_labels(team_game_stats=data['team_stats']))
    sw = shapiro_wilk(distributions=fields)
    norms = normality_filter(shapiro_wilk=sw, threshold=SW_NORM_THRESHOLD)
    
    return norms

def z_scores(**kwargs):
    data = kwargs['data']

    avg = np.average(data)
    stddev = np.std(data)

    return map(lambda d: (d - avg) / stddev, data)
