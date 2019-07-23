import matplotlib.pyplot as plt
import estimator as est
import team_game_statistics as tgs
from scipy.stats import shapiro
import numpy as np

#There seems to be a correlation between the statistic value of the Shapiro-Wilk test and the 
#"normality" of the given distribution. We will say that any distribution with a statistic 
#greater than or equal to .8 is a failure to reject H0. Also, the p values are incredibly 
#low for all tests, but I am unsure why at this point.
SW_NORM_THRESHOLD = 0.80

def reorder(**kwargs):
    """Orders features such that -0 feature follows -1 feature

    Args:
         features: map of features to values
    
    Returns: tuple with two entries, where the first is the ordered features
             and the second is any features where a matching feature could 
             not be found
    """
    fs = kwargs['features']

    res = []
    no_match = []
    for field in filter(lambda x: '-0' in x, set(fs.keys())):
        lis = filter(lambda e: e[0 : len(e) - 2] == field[0 : len(field) - 2], 
                     filter(lambda x: '-1' in x, set(fs.keys())))
        if len(lis) == 1:
            res.append((field, fs[field]))
            res.append((lis[0], fs[lis[0]]))
        else:
            no_match.append(field)

    return res, no_match

def histogram(**kwargs):
    """Draws a histogram plot for each field in avgs

    Args: 
         avgs: Averages for each team in each game

    Returns: None
    """
    avgs, team_stats = kwargs['avgs'], kwargs['team_stats']

    fields, _ = est.input_data(game_averages=avgs, labels=tgs.add_labels(team_game_stats=team_stats))

    data, _ = reorder(features=fields)

    for i in range(1, len(data), 2):
        plt.figure(num=i)
        plt.hist(data[i - 1][1], bins='auto')
        plt.title(data[i - 1][0])

        plt.hist(data[i][1], bins='auto')
        plt.title(data[i][0])

    plt.show()

def shapiro_wilk(**kwargs):
    """Computes the shapiro wilk statistical test for a given map of distributions

    Args:
         distributions: map of field name to values where values, in most calling 
         contexts, are the averages
   
    Returns: map of field name to shapiro wilk score
    """
    dist = kwargs['distributions']

    result = {}
    for k, d in dist.iteritems():
        result[k] = shapiro(d)
    
    return result

def similar_field(**kwargs):
    """Given a field and a collection of fields, finds the corresponding
       field by checking equality of the first n - 2 characters

    Args:
         field: field to search for
         all_fields: collection of fields to search within
    
    Returns: the first matching field if one is found, otherwise None
    """
    f, afs = kwargs['field'], kwargs['all_fields']

    res = filter(lambda e: e[0 : len(e) - 2] == f[0 : len(f) - 2], afs)
    return res[0] if res else None

def normality_filter(**kwargs):
    """Given shapiro wilk scores and a threshold, filters distributions
       according to their shaprio wilk statistical value being greater 
       than or equal to the given threshold. Note this function will 
       opt a field in if its corresponding field passes the threshold.

    Args:
         shaprio_wilk: map of fields to their shapiro_wilk scores
         threshold: value for comparing shapiro wilk statistical 
                    scores

    Returns: map of fields to their shapiro wilk scores
    """
    sw, th = kwargs['shapiro_wilk'], kwargs['threshold']
    
    result = {}
    for f, val in filter(lambda i: '-0' in i[0], set(sw.iteritems())):
        sf = similar_field(field=f, all_fields=filter(lambda k: '-1' in k, set(sw.keys())))
        if sf and (sw[sf][0] >= th or val[0] >= th):
            result[f] = val
            result[sf] = sw[sf]    
    return result

def normal_dists(**kwargs):
    """Computes shapiro wilk scores for a given set of averages 
       or distributions and returns a filtered set of those distributions
       by their shapiro wilk scores

    Args: 
         field_avgs: map of field name to its average values
    
    Returns: map of fields to their shapiro wilk scores
    """
    f_avgs = kwargs['field_avgs']
        
    sw = shapiro_wilk(distributions=f_avgs)
    norms = normality_filter(shapiro_wilk=sw, threshold=SW_NORM_THRESHOLD)
    
    return norms

def z_scores_args(**kwargs):
    """Computes zscores for given data, based off of a given mean and
       stddev

    Args:
         data: input values
         mean: mean of input values
         stddev: stddev of input values

    Returns: zscores of data
    """
    data, mean, stddev = kwargs['data'], kwargs['mean'], kwargs['stddev']
    
    return map(lambda d: (d - mean) / stddev, data)

def z_scores(**kwargs):
    """Computes zscores for given data, based off of a mean and stddev
       derived from data 

    Args:
         data: input values

    Returns: zscores of data
    """
    data = kwargs['data']

    return z_scores_args(**{'data': data, 'mean': np.average(data), 'stddev': np.std(data)})

def reverse_zscores(**kwargs):
    """Reverses zscores for given data, based off of a given mean and
       stddev

    Args:
         data: input values
         mean: mean of input values
         stddev: stddev of input values

    Returns: original values of data
    """
    data, mean, stddev = kwargs['data'], kwargs['mean'], kwargs['stddev']
    
    return map(lambda v: v * stddev + mean, data)
