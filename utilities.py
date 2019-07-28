import csv
import glob
import os.path as path
import pickle
import os

DATA_CACHE_DIR = 'data_cache'
FIGURE_DIR = 'figures'

#functions that serve a general purpose

def read_file(**kwargs):
    """Passes a list of values to a given function, saves the result 
       and returns it

    Args:
         input_file: csv file with data
         func: function to map the list of values to something else
    
    Returns: result of func accepting list of values
    """
    input_file, func = kwargs['input_file'], kwargs['func']
    result = None

    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        result = func(csv_reader=csv_reader)
    return result

def convert_type(**kwargs):
    """Casts a value to a given type, returns value
       if unsuccessful

    Args:
        types: collection of types, first type to succeed 
               will be resulting type
        value: a value
    
    Returns: value
    """
    types, value = kwargs['types'], kwargs['value']

    for t in types:
        try:
            return t(value)
        except ValueError:
            pass
    return value

def subset_of_map(**kwargs):
    """Given a map and a set of keys, returns a map with the 
       specified keys removed

    Args:
         full_map: a map
         take_out_keys: keys to remove from map
    
    Returns: a map
    """
    full_map, take_out_keys = kwargs['full_map'], kwargs['take_out_keys']

    return {k: full_map[k] for k in set(full_map.keys()).difference(take_out_keys)}

def merge_maps(**kwargs):
    """Given two maps, merges them with the specified merge_op

    Args:
         map1: a map
         map2: a map
         merge_op: an operator to merge map1 and map2

    Returns: a map

    Raises: 
           ValueError: if map1.keys() != map2.keys()
    """
    map1, map2, merge_op = kwargs['map1'], kwargs['map2'], kwargs['merge_op']
    
    if set(map1.keys()) == set(map2.keys()):
        return {k: merge_op(map1[k], map2[k]) for k in map1.keys()}
    else:
        raise ValueError("set(map1.keys()) == set(map2.keys()) must be true, diff: (%s)" % 
                         (str(set(map1.keys()).difference(map2.keys()))))

def create_map(**kwargs):
    """Creates a map with default values given keys and a value

    Args:
         keys: set of keys for the map
         default: value for all keys
    
    Returns: a map
    """
    keys, default = kwargs['keys'], kwargs['default']

    return {k: default for k in keys}

def write_to_cache(**kwargs):
    """Writes given data to a given cache directory.

    Args:
         cache_dir: directory to write to
         data: data to write to cache
    
    Returns: None
    """
    cd, data = kwargs['cache_dir'], kwargs['data']

    for sea, d in data.iteritems():
        with open(path.join(path.abspath(cd), path.basename(sea)), 'wb') as fh:
            pickle.dump(d, fh)

def read_from_cache(**kwargs):
    """Reads from a given cache directory and renames resulting entries per
       context_dir argument.

    Args:
         cache_dir: directory to read from
         context_dir: directory to serve as parent of names in resulting
                      key set
    
    Returns: map of file names to data
    """
    cd, context = kwargs['cache_dir'], kwargs['context_dir']

    result = {}
    for sea in glob.glob(path.join(path.abspath(cd), '*')):
        with open(sea, 'rb') as fh:
            entry = path.join(context, path.basename(sea))
            result[entry] = pickle.load(fh)
            
    return result

def model_data(**kwargs):
    """Reads from cache, figures out which seasons need to be computed, computes
       needed data, writes computed data to cache, and merges the data read from 
       cache and computed in memory. Will create cache if it does not exist.

    Args:
         cache_dir: directory of cache
         all_dirs: collection of directories
         cache_read_func: function to read from cache
         cache_write_func: functino to write from cache
         comp_func: function to compute data over seasons, must accept at least
                    'dirs' as argument
         comp_func_args: extra arguments to comp_func
         context_dir: parent path for entires in cache, used by cache_read_func

    Returns: tuple of map of data and seasons that were computed in memory
    """
    cd, crf, cwf, ad, cf, cfa, ctx = kwargs['cache_dir'], kwargs['cache_read_func'], kwargs['cache_write_func'],\
                                kwargs['all_dirs'], kwargs['comp_func'], kwargs['comp_func_args'], kwargs['context_dir']

    if not path.exists(cd):
        os.makedirs(cd)
    cache = crf(cache_dir=cd, context_dir=path.abspath(ctx))
    compute_seasons = set(ad).difference(set(cache.keys()))
    data = cf(dirs=compute_seasons, **cfa)
    cwf(cache_dir=cd, data=data)

    return {k: data[k] if k in data else cache[k] for k in set(ad)}, compute_seasons

def print_collection(**kwargs):
    """Prints a given collection to stdout.

    Args:
         coll: input collection
    
    Returns: None
    """
    coll = kwargs['coll']

    for e in coll:
        print(e)

def print_cache_reads(**kwargs):
    """Prints a given collection to stdout.

    Args:
         coll: input collection
         data_origin: origin of data (e.g. location on disk)
    
    Returns: None
    """
    cs, do = kwargs['coll'], kwargs['data_origin']

    print("\n")
    print("Done reading data from disk. Computed data for seasons: ")
    if len(cs) > 0:
        print_collection(coll=cs)
    else:
        print("All data was read from %s" % (path.abspath(do)))
    print("\n")
