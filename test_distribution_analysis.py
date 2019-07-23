import distribution_analysis as da

def test_reorder():
    field = 'field'
    in_data = {field + '1-1': 1, field + '1-0': 2, field + '2-1': 3, field + '2-0': 4, 
               field + '3-0': 1}
    out, nm = da.reorder(features=in_data)

    end = len(field) + 1
    for i in range(0, len(out), 2):
        assert out[i][0][0 : end] == out[i + 1][0][0 : end]

    assert 1 == len(nm)
        
