
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

def test_id_maker():
    id_maker = util.IDMaker(
            'map_name/episode/agent/frame',
            prefixes={
                'episode':  'ep',
                'agent':    'agent',
                'frame':    'frame'},
            format_spec={
                'episode':  '03d',
                'agent':    '03d',
                'frame':    '08d'})

    id_preprocess = [
            util.AttrDict(map_name='Town01', episode=1, agent=1, frame=1000),
            util.AttrDict(map_name='Town01', episode=1, agent=2, frame=1000),
            util.AttrDict(map_name='Town01', episode=1, agent=3, frame=1000),
            util.AttrDict(map_name='Town01', episode=1, agent=1, frame=2000),
            util.AttrDict(map_name='Town01', episode=1, agent=2, frame=2000),
            util.AttrDict(map_name='Town01', episode=2, agent=1, frame=500),
            util.AttrDict(map_name='Town01', episode=2, agent=2, frame=500),
            util.AttrDict(map_name='Town01', episode=2, agent=3, frame=530),
            util.AttrDict(map_name='Town01', episode=2, agent=4, frame=530),
            util.AttrDict(map_name='Town02', episode=3, agent=1, frame=100),
            util.AttrDict(map_name='Town02', episode=3, agent=2, frame=100),
            util.AttrDict(map_name='Town02', episode=4, agent=1, frame=100),
            util.AttrDict(map_name='Town02', episode=4, agent=2, frame=100),]

    ids = ['Town01/ep001/agent001/frame00001000',
            'Town01/ep001/agent002/frame00001000',
            'Town01/ep001/agent003/frame00001000',
            'Town01/ep001/agent001/frame00002000',
            'Town01/ep001/agent002/frame00002000',
            'Town01/ep002/agent001/frame00000500',
            'Town01/ep002/agent002/frame00000500',
            'Town01/ep002/agent003/frame00000530',
            'Town01/ep002/agent004/frame00000530',
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent001/frame00000100',
            'Town02/ep004/agent002/frame00000100',]

    expected = {'map_name': 0, 'episode': 1, 'agent': 2, 'frame':3}
    actual = id_maker.sample_pattern
    assert actual == expected

    expected = '{map_name:}/ep{episode:03d}/agent{agent:03d}/frame{frame:08d}'
    actual = id_maker.fstring
    assert actual == expected

    def f(d):
        return id_maker.make_id(
                map_name=d.map_name, episode=d.episode,
                agent=d.agent, frame=d.frame)
    actual = util.map_to_list(f, id_preprocess)
    expected = ids
    assert actual == expected

def test_filter_ids():
    id_maker = util.IDMaker('map_name/episode/agent/frame')
    ids = ['Town01/ep001/agent001/frame00001000',
            'Town01/ep001/agent002/frame00001000',
            'Town01/ep001/agent003/frame00001000',
            'Town01/ep001/agent001/frame00002000',
            'Town01/ep001/agent002/frame00002000',
            'Town01/ep002/agent001/frame00000500',
            'Town01/ep002/agent002/frame00000500',
            'Town01/ep002/agent003/frame00000530',
            'Town02/ep002/agent004/frame00000530',
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent001/frame00000100',
            'Town02/ep004/agent002/frame00000100',
            'Town03/ep005/agent001/frame00001000',
            'Town03/ep005/agent001/frame00002000',]
    random.shuffle(ids)

    filter = {'map_name': 'Town02'}
    actual = id_maker.filter_ids(ids, filter)
    actual.sort()
    expected = [
            'Town02/ep002/agent004/frame00000530',
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent001/frame00000100',
            'Town02/ep004/agent002/frame00000100',]
    expected.sort()
    assert actual == expected

    filter = {'map_name': 'Town02', 'episode': 'ep003'}
    actual = id_maker.filter_ids(ids, filter)
    actual.sort()
    expected = [
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',]
    expected.sort()
    assert actual == expected
    
    filter = {'agent': 'agent001', 'episode': 'ep002'}
    actual = id_maker.filter_ids(ids, filter, inclusive=False)
    actual.sort()
    expected = [
            'Town01/ep001/agent002/frame00001000',
            'Town01/ep001/agent003/frame00001000',
            'Town01/ep001/agent002/frame00002000',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent002/frame00000100',]
    expected.sort()
    assert actual == expected

def test_group_ids():
    id_maker = util.IDMaker('annotation/subtype/slide/patch_size/magnification/coordinate')
    patch_ids = [
        'Stroma/MMRd/VOA-1000A/512/20/0_0',
        'Stroma/MMRd/VOA-1000A/512/20/2_2',
        'Stroma/MMRd/VOA-1000A/512/10/0_0',
        'Stroma/MMRd/VOA-1000A/256/10/0_0',
        'Tumor/POLE/VOA-1000B/256/10/0_0']
    
    groups, labels = id_maker.group_ids(patch_ids, ['patch_size'])
    
    expected = {'patch_size': ['256', '512']}
    util.sort_nested_dict_of_list(expected)
    actual = labels
    util.sort_nested_dict_of_list(actual)
    assert actual == expected

    expected = {
        '512': [
            'Stroma/MMRd/VOA-1000A/512/20/0_0',
            'Stroma/MMRd/VOA-1000A/512/10/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/2_2',
        ],
        '256': [
            'Stroma/MMRd/VOA-1000A/256/10/0_0',
            'Tumor/POLE/VOA-1000B/256/10/0_0'
        ]
    }
    util.sort_nested_dict_of_list(expected)
    actual = groups
    util.sort_nested_dict_of_list(actual)
    assert actual == expected
    
    groups, labels = id_maker.group_ids(patch_ids, ['patch_size', 'magnification'])

    expected = {'patch_size': ['256', '512'], 'magnification': ['10', '20']}
    util.sort_nested_dict_of_list(expected)
    actual = labels
    util.sort_nested_dict_of_list(actual)
    assert actual == expected


    expected = {
        '512': {
            '20': [
                'Stroma/MMRd/VOA-1000A/512/20/0_0',
                'Stroma/MMRd/VOA-1000A/512/20/2_2',
            ],
            '10': [
                'Stroma/MMRd/VOA-1000A/512/10/0_0',
            ]
        },
        '256': {
            '20': [ ],
            '10': [
                'Stroma/MMRd/VOA-1000A/256/10/0_0',
                'Tumor/POLE/VOA-1000B/256/10/0_0'
            ]
        }
    }
    util.sort_nested_dict_of_list(expected)
    actual = groups
    util.sort_nested_dict_of_list(actual)
    assert actual == expected


def test_group_ids_by_index():
    id_maker = util.IDMaker('annotation/subtype/slide/patch_size/magnification/coordinate')
    patch_ids = [
        'Stroma/MMRd/VOA-1000A/512/20/0_0',
        'Stroma/MMRd/VOA-1000A/512/20/2_2',
        'Stroma/MMRd/VOA-1000A/512/10/0_0',
        'Stroma/MMRd/VOA-1000A/256/20/0_0',
        'Stroma/MMRd/VOA-1000A/256/10/0_0',
        'Tumor/POLE/VOA-1000B/256/10/0_0']
    
    actual = id_maker.group_ids_by_index(patch_ids,
            include=['annotation', 'magnification'])
    util.sort_nested_dict_of_list(actual)
    expected = {
        'Stroma/20': [
            'Stroma/MMRd/VOA-1000A/256/20/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/2_2'
        ],
        'Stroma/10': [
            'Stroma/MMRd/VOA-1000A/256/10/0_0',
            'Stroma/MMRd/VOA-1000A/512/10/0_0'
        ],
        'Tumor/10': [
            'Tumor/POLE/VOA-1000B/256/10/0_0'
        ]
    }
    util.sort_nested_dict_of_list(expected)
    assert actual == expected
    
    actual = id_maker.group_ids_by_index(patch_ids,
            exclude=['slide', 'magnification'])
    util.sort_nested_dict_of_list(actual)
    expected = {
        'Stroma/MMRd/512/0_0': [
            'Stroma/MMRd/VOA-1000A/512/10/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/0_0'
        ],
        'Stroma/MMRd/512/2_2': [
            'Stroma/MMRd/VOA-1000A/512/20/2_2'
        ],
        'Stroma/MMRd/256/0_0': [
            'Stroma/MMRd/VOA-1000A/256/10/0_0',
            'Stroma/MMRd/VOA-1000A/256/20/0_0'
        ],
        'Tumor/POLE/256/0_0': [
            'Tumor/POLE/VOA-1000B/256/10/0_0'
        ]
    }
    util.sort_nested_dict_of_list(expected)
    assert actual == expected
