import os
import copy
import random

import numpy as np
import pytest

import utility as util

random.seed(256)

def sort_list_of_list(ll):
    for l in ll:
        l.sort()

def test_MANY():
    """Test many small functions
    """
    # test util.space_list
    expected = "1 2 3 4 5 6"
    actual  = util.space_list([1, 2, 3, 4, 5, 6])
    assert expected == actual

    # test util.count
    assert 2 == util.count(lambda x: x == True,
            [True, False, True, False, False])
    
    # test util.merge_list_of_list
    expected = [1, 2, 3, 4, 5, 6]
    actual = util.merge_list_of_list([[1, 2], [3, 4], [5, 6]])
    assert expected == actual

    data = {'a': {'b': [3,1,2]}, 'c': [3,4,1,2], 'd': dict()}

    expected = {'a': {'b': [1,2,3]}, 'c': [1,2,3,4], 'd': dict()}
    actual = copy.deepcopy(data)
    util.sort_nested_dict_of_list(actual)
    assert actual == expected

    expected = {'a': {'b': [3,2,1]}, 'c': [4,3,2,1], 'd': dict()}
    actual = copy.deepcopy(data)
    util.sort_nested_dict_of_list(actual, reverse=True)
    assert actual == expected

    # test util.inner_keys_from_nested_dict
    d = {
        'a': {
            'a1': {
                'a11': 11,
                'a12': 12,
            },
            'a2': {
                'a21': 21,
            },
            'a3': 3,
        },
        'b': {
            'a1': 1,
            'b2': {
                'b21': 21,
                'b22': 22,
                'b23': 23,
            },
        },
        'c': {
            'c1': 1,
        },
    }
    expected = [
            ['a', 'b', 'c'],
            ['c1', 'a1', 'b2', 'a1', 'a2', 'a3'],
            ['a21', 'a11', 'a12', 'b21', 'b22', 'b23']]
    sort_list_of_list(expected)

    actual = util.inner_keys_from_nested_dict(d, layers=1)
    sort_list_of_list(actual)
    assert actual == expected[:1]

    actual = util.inner_keys_from_nested_dict(d, layers=2)
    sort_list_of_list(actual)
    assert actual == expected[:2]

    actual = util.inner_keys_from_nested_dict(d, layers=3)
    sort_list_of_list(actual)
    assert actual == expected

    actual = util.inner_keys_from_nested_dict(d, layers=4)
    sort_list_of_list(actual)
    assert actual == expected

@pytest.mark.parametrize(
    "seq,expected_subseq,expected_slice,expected_size",
    [
        pytest.param([1,2,3], [1,2,3], slice(0, 3), 3),
        pytest.param([1,2,1,2,3], [1,2,3], slice(2, 5), 3),
        pytest.param([2,1,1,2,3], [1,2,3], slice(2, 5), 3),
        pytest.param([1,2,3,1,2], [1,2,3], slice(0, 3), 3),
        pytest.param([1,2,3,3,2], [1,2,3], slice(0, 3), 3),
        pytest.param([1,2,3,5,6], [1,2,3], slice(0, 3), 3),
        pytest.param([1,2,4,5,6], [4,5,6], slice(2, 5), 3),
        pytest.param([1,2,4,5,6], [4,5,6], slice(2, 5), 3),
    ]
)
def test_subsequences(seq, expected_subseq, expected_slice, expected_size):
    _slice, size = util.longest_consecutive_increasing_subsequence(seq)
    subseq = seq[_slice]
    assert subseq == expected_subseq
    assert _slice == expected_slice
    assert size == expected_size

def test_subsequences_2():
    seq =  np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64)
    _slice, size = util.longest_consecutive_increasing_subsequence(seq)
    print(size, _slice, seq[_slice])
