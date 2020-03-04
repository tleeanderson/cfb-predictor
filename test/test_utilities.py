import utilities as util
import operator as op

def test_read_file():
    in_file = 'file_that_doesnt_exist'

    def file_reader(**kwargs):
        csv = kwargs['csv_reader']
        return None

    try:
        out = util.read_file(input_file=in_file, func=file_reader)
    except IOError:
        assert(True)
    else:
        assert(False)

def test_convert_type():
    types, bad_val, good_val = (int, float), 'f', '1'

    out = util.convert_type(types=types, value=good_val)
    bad_out = util.convert_type(types=types, value=bad_val)
    assert(out == 1)
    assert(bad_out == bad_val)

def test_subset_of_map():
    in_data, take_out = {i: i for i in range(10)}, {i for i in range(5)}

    out = util.subset_of_map(full_map=in_data, take_out_keys=take_out)

    assert(set(in_data.keys()).difference(take_out) == set(out.keys()))

def test_merge_maps():
    m1 = {i: i for i in range(1, 2)}
    m2 = {i: i for i in list(m1.keys())}

    out_add = util.merge_maps(map1=m1, map2=m2, merge_op=op.add)
    out_sub = util.merge_maps(map1=m1, map2=m2, merge_op=op.sub)
    m3 = {i: i for i in set(m1.keys()).union({45})}
    
    assert(list(out_add.items())[0][1] == 2)
    assert(list(out_sub.items())[0][1] == 0)
    try:
        util.merge_maps(map1=m1, map2=m3, merge_op=op.add)
    except ValueError:
        assert(True)
    else:
        assert(False)

def test_create_map():
    nv = 3
    keys, default = {i for i in range(nv)}, 'val'

    out = util.create_map(keys=keys, default=default)
    assert(len(out) == nv)
    assert([v == default for v in list(out.values())])
    

    
