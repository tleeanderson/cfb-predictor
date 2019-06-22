import csv

#functions that serve a general purpose

def read_file(**kwargs):
    input_file, func = kwargs['input_file'], kwargs['func']
    result = None

    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        result = func(csv_reader=csv_reader)
    return result

def convert_type(**kwargs):
    types, value = kwargs['types'], kwargs['value']

    for t in types:
        try:
            return t(value)
        except ValueError:
            pass
    raise ValueError("Could not convert %s" % (value))

def subset_of_map(**kwargs):
    full_map, take_out_keys = kwargs['full_map'], kwargs['take_out_keys']

    return {k: full_map[k] for k in set(full_map.keys()).difference(take_out_keys)}

def merge_maps(**kwargs):
    map1, map2, merge_op = kwargs['map1'], kwargs['map2'], kwargs['merge_op']
    
    if set(map1.keys()) == set(map2.keys()):
        return {k: merge_op(map1[k], map2[k]) for k in map1.keys()}
    else:
        raise ValueError("set(map1.keys()) == set(map2.keys()) must be true, diff: (%s)" % 
                         (str(set(map1.keys()).difference(map2.keys()))))

def create_map(**kwargs):
    keys, default = kwargs['keys'], kwargs['default']

    return {k: default for k in keys}
