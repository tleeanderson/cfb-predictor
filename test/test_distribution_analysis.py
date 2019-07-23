import distribution_analysis as da
import numpy as np

def test_reorder():
    field = 'field'
    in_data = {field + '1-1': 1, field + '1-0': 2, field + '2-1': 3, field + '2-0': 4, 
               field + '3-0': 1}
    out, nm = da.reorder(features=in_data)

    end = len(field) + 1
    for i in range(0, len(out), 2):
        assert out[i][0][0 : end] == out[i + 1][0][0 : end]

    assert 1 == len(nm)
    
def test_similar_field():
    f = 'field'
    in_data = {f + '1-1', f + '2-0', f + '2-1'}
    val = f + '1-1'
    out = da.similar_field(field=val, all_fields=in_data)

    assert out == val
    assert da.similar_field(field=f + '3-1', all_fields=in_data) is None

def test_normality_filter():
    nn_field, norm = 'non_normal_field', 'normal'
    nn_field0, nn_field1, norm0, norm1 = nn_field + '-0', nn_field + '-1',\
                                         norm + '-0', norm + '-1'
    nn_dist, norm_dist = np.random.lognormal(0, 2, 1000), np.random.normal(0, 1, 1000)

    #test opting in of fields whose corresponding field falls below threshold
    in_data = {nn_field0: nn_dist, nn_field1: nn_dist, norm1: norm_dist, norm0: nn_dist}
    sw = da.shapiro_wilk(distributions=in_data)
    out = da.normality_filter(shapiro_wilk=sw, threshold=0.98)

    assert len(out) == 2
    for entry in out.iteritems():
        assert norm in entry[0]

def test_z_scores():
    mean, stddev, nv = 0, 1, 10000
    data = list(np.random.normal(mean, stddev, nv))

    out_zsa = da.reverse_zscores(data=da.z_scores_args(data=data, mean=mean, stddev=stddev), 
                             mean=mean, stddev=stddev)
    
    assert list(out_zsa) == data
    
