import csv

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
    """Converts a value to a given type, returns value error 
       if unsuccessful

    Args:
        types: collection of types, first type to succeed 
               will be resulting type
        value: a value
    
    Returns: value
    
    Raises: 
           ValueError: if value cannot be casted to one of
                       the given types
    """
    types, value = kwargs['types'], kwargs['value']

    for t in types:
        try:
            return t(value)
        except ValueError:
            pass
    raise ValueError("Could not convert %s" % (value))

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
