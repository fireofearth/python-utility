import os
import json
from pathlib import Path
import math
import random
import operator
import collections
import itertools
import functools

import numpy as np

class UtilityException(Exception):
    pass

def hello_world():
    """Test that package installation works"""
    return "Hello World"

####################
# General operations
####################

def classname(x):
    return type(x).__name__

#################
# File operations
#################

def get_dirname_of(fn):
    """Get absolute path of the immediate directory the file is in
    Parameters
    ----------
    fn : str
        The path i.e. /path/to/image.png
    Returns
    -------
    str
        The dirname i.e /path/to
    """
    return os.path.dirname(os.path.abspath(fn))

def save_json(filepath, obj):
    """Serialize object as a JSON formatted stream and save to a file.

    Parameters
    ----------
    filepath : str
        Path of file to save JSON.
    obj : dict
        Object to serialize.
    """
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def load_json(filepath):
    """Load JSON file to a Python object.

    Parameters
    ----------
    filepath : str
        Path of file to load JSON.
    """
    with open(filepath) as f:
        return json.load(f)

def strip_extension(path):
    """Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))

###########################
# Not-functional operations
###########################

def do_on_nested_dict_of_list(f, dl, *args, **kwargs):
    if isinstance(dl, list):
        f(dl, *args, **kwargs)
    elif isinstance(dl, dict):
        for v in dl.values():
            sort_nested_dict_of_list(v, *args, **kwargs)
    else:
        pass

def sort_nested_dict_of_list(dl, **kwargs):
    """Sort lists contained inside nested dict.
    Optional arguments for list.sort() are passed on as arguments."""
    def f(l, **kwargs):
        l.sort(**kwargs)
    do_on_nested_dict_of_list(f, dl, **kwargs)

def shuffle_nested_dict_of_list(dl):
    """Shuffle lists contained inside nested dict."""
    def f(l):
        random.shuffle(l)
    do_on_nested_dict_of_list(f, dl)

#########################
# (Functional) operations
#########################

def range_to_list(*args):
    """Creates a range and converts it to a list."""
    return list(range(*args))

def map_to_list(f, l):
    """Does map operation and then converts map object to a list."""
    return list(map(f, l))

def map_to_ndarray(f, l):
    """Does map operation and then converts map object to a list."""
    return np.array(map_to_list(f, l))

def zip_to_list(*args):
    """Does a zip operation and then converts zip object to a list."""
    return list(zip(*args))

def filter_to_list(f, l):
    """Filter from list elements that return true under f(), returning a list.
    Example: (lambda x: x > 2, [1,2,3,4,5]) -> [3,4,5]"""
    return list(filter(f, l))

def compress_to_list(l, bl):
    """Filter list using a list of selectors, returning a list.
    Example: ([1,2,3,4,5], [True, False, False, True, False]) -> [1,4]"""
    # TODO: DEPRECATED
    return list(itertools.compress(l, bl))

def reduce(*args, **kwargs):
    """

    Parameters
    ==========
    f : (function v, acc: f(v, acc))
    l : iterable
    i : any (optional)
    """
    return functools.reduce(*args, **kwargs)

def count(f, l):
    """Count the number of occurances in a list.
    Parameters
    ==========
    f : function
        Function that maps element in list to bool
    
    l : list
        List to count occurances
    
    Returns
    =======
    int
        Number of occurances
    """
    return len(filter_to_list(f, l))

def merge_list_of_list(ll):
    """Concatenate iterable of iterables into one list."""
    return list(itertools.chain.from_iterable(ll))

def space_list(l):
    return ' '.join(map(str, l))

def underscore_list(l):
    return space_list(l).replace(' ', '_')

def reverse_list(l):
    return list(reversed(l))

def pairwise(l):
    """Make a list of consecutive pairs given a list. 
    Example: [1,2,3] -> [(1,2),(2,3)]"""
    a, b = itertools.tee(l)
    next(b, None)
    return list(zip(a, b))

def pairwise_do(f, l):
    """Make a list by applying operation on consecutive pairs in a list.
    Example: [1,2,3] -> [1+2,2+3]
    """
    a, b = itertools.tee(l)
    next(b, None)
    return [f(i, j) for i, j in zip(a, b)]

def unzip(ll):
    """The inverse operation to zip.
    Example: [('a', 1), ('b', 2), ('c', 3)] -> [('a', 'b', 'c'), (1, 2, 3)]
    """
    return list(zip(*ll))

def longest_subsequence(l, cond=None):
    """Get the longest subsequence of the list composed of entries
    where cond is true. For example:
    (lambda l, i: l[i], [1, 0, 1, 1, 1, 0, 1]) -> ((slice(2, 5), 3)
    
    Parameters
    ==========
    l : list or np.array
        The list to get subsequence from. 
    cond : function (l, i) -> bool
        To check whether list entries belong to the subsequence.
        If not passed the check using truthiness. 
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
        Check that such subsequence exist using the length.
    """
    if not cond:
        cond = lambda x: x
    n = len(l)
    longest_size  = 0
    longest_begin = None
    longest_end   = None
    curr_end = 0
    while curr_end < n:
        if cond(l[curr_end]):
            curr_size = 1
            curr_begin = curr_end
            while curr_end < n - 1 and cond(l[curr_end+1]):
                curr_size += 1
                curr_end  += 1
            if curr_size > longest_size:
                longest_size = curr_size
                longest_begin = curr_begin
                longest_end = curr_end
            curr_end += 1
        curr_end += 1
    if longest_size == 0:
        return None, longest_size
    else:
        return slice(longest_begin, longest_end+1), longest_size

def longest_sequence_using_split(l, split):
    """Get the longest subsequence of the list after it has been
    split into segments. For example if split is the function
    lambda l, i: l[i + 1] != l[i] + 1 if i < len(l) - 1 else False
    then
    ([1, 2, 3, 2, 3, 4, 5, 1, 2], split) -> ((slice(3, 7), 4)
    as split gives
    [1, 2, 3 | 2, 3, 4, 5 | 1, 2]
    
    Parameters
    ==========
    l : list or np.array
        The list to get subsequence.
    split : function (l, i) -> bool
        To check whether to split the list.
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
    """
    n = len(l)
    longest_size  = 0
    longest_begin = None
    longest_end   = None
    curr_begin = 0
    curr_end   = 0
    while curr_end < n:
        if split(l, curr_end) or curr_end == n - 1:
            if (curr_end - curr_begin + 1)  > longest_size:
                longest_size  = curr_end - curr_begin + 1
                longest_begin = curr_begin
                longest_end   = curr_end
            curr_begin = curr_end + 1
            curr_end   = curr_end + 1
        else:
            curr_end  += 1
    if longest_size == 0:
        return None, longest_size
    else:
        return slice(longest_begin, longest_end+1), longest_size

def longest_consecutive_increasing_subsequence(l):
    """Get the longest consecutively increasing subsequence.
    
    Parameters
    ==========
    list of int
        The list to get subsequence.
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
    """
    def split(l, i):
        try:
            return l[i + 1] != l[i] + 1
        except:
            return False
    return longest_sequence_using_split(l, split)

def longest_consecutive_decreasing_subsequence(l):
    """Get the longest consecutively decreasing subsequence.
    
    Parameters
    ==========
    list of int
        The list to get subsequence.
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
    """
    def split(l, i):
        try:
            return l[i + 1] != l[i] - 1
        except:
            return False
    return longest_sequence_using_split(l, split)

def inner_keys_from_nested_dict(d, layers=2):
    """Get the inner keys of a of nested of dict. Example:
    ({'a': {'a1': 1 }, 'b': { 'b1': 1, 'b2': 2, }}, 2) -> [['a', 'b', 'c'], ['a1', 'b1', 'b2']]

    Parameters
    ==========
    d : dict
        Nested dict to get keys from.
    layers : int
        Number of layers of inner keys to extract.
        If layers=1 then equivalent to [list(d.keys())].

    Returns
    =======
    list of list
        Each entry in the list is collection of keys at at that level of the nested dict.
        The collection of keys corresponding to the level may be duplicated.
    """
    ll = []
    vl = [d]
    for _ in range(layers):
        l = []
        f = lambda x: isinstance(x, dict)
        q = collections.deque(filter_to_list(f, vl))
        vl = []
        if not q:
            break
        while q:
            d = q.pop()
            keys, values = unzip(d.items())
            vl.append(values)
            l.append(keys)
        l = merge_list_of_list(l)
        vl = merge_list_of_list(vl)
        ll.append(l)
    return ll

# itertools.* functions should output to list
# Replacements of the *_to_list() functions

def compress(*args, **kwargs):
    """Filter list using a list of selectors, returning a list.
    Example: ([1,2,3,4,5], [True, False, False, True, False]) -> [1,4]"""
    return list(itertools.compress(*args, **kwargs))

def accumulate(*args, **kwargs):
    """Accumulate list, returning a list.
    Example: [1,2,3] -> [1,3,6]; [1,2,3], initial=100 -> [100,101,103,106]"""
    return list(itertools.accumulate(*args, **kwargs))

#################
# Math operations
#################

def sgn(x):
    """Get the sign of a number as int. 1.2 -> 1 and -1.2 -> -1"""
    return int(math.copysign(1, x))

#######################
# Numpy math operations
#######################

def is_positive_semidefinite(X):
    """Check that a matrix is positive semidefinite
    
    Based on:
    https://stackoverflow.com/a/63911811
    """
    if X.shape[0] != X.shape[1]:
        return False
    if not np.all( X - X.T == 0 ):
        return False
    try:
        regularized_X = X + np.eye(X.shape[0]) * 1e-14
        np.linalg.cholesky(regularized_X)
    except np.linalg.linalg.LinAlgError as err:
        if "Matrix is not positive definite"  == str(err):
            return False
        raise err
    return True

def is_positive_definite(X):
    """Check that a matrix is positive definite
    
    Based on:
    https://stackoverflow.com/a/63911811
    """
    if X.shape[0] != X.shape[1]:
        return False
    if not np.all( X - X.T == 0 ):
        return False
    try:
        np.linalg.cholesky(X)
    except np.linalg.linalg.LinAlgError as err:
        if "Matrix is not positive definite"  == str(err):
            return False
        raise err
    return True

def reflect_radians_about_x_axis(r):
    r = (-r) % (2*np.pi)
    return r

def reflect_radians_about_y_axis(r):
    r = (r + np.pi) % (2*np.pi)
    return r

def determinant_2d(A):
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    return a*d - b*c

def inverse_2d(A):
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    return 1. / (a*d - b*c) * np.array([[d, -b], [-c, a]])

def rotation_2d(theta):
    """2D rotation matrix. If x is a column vector of 2D points then
    `rotation_2d(theta) @ x` gives the rotated points.
    
    Parameters
    ==========
    theta : float
        Rotates points clockwise about origin if theta is positive.
        Rotates points counter-clockwise about origin if theta is negative
    
    Returns
    =======
    np.array
        2D rotation matrix of shape (2, 2).
    """
    return np.array([
            [ np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])

def distances_from_line_2d(points, x_start, y_start, x_end, y_end):
    """Get the distances from each point to a line spanned by line segment from
    (x_start, y_start) to (x_end, y_end). Works for horizontal and vertical lines.
    
    Based on:
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameters
    ==========
    points : np.array or list
        One 2D point, or multiple 2D points of shape (n, 2).
    x_start : float
        Line segment component
    y_start : float
        Line segment component
    x_end : float
        Line segment component
    y_end : float
        Line segment component

    Returns
    =======
    float or np.array
        Distance of point to line, or array of distances from points to line.
    """
    points = np.array(points)
    if points.ndim == 1:
        return np.abs((x_end - x_start)*(y_start - points[1]) - (x_start - points[0])*(y_end - y_start)) \
                / np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    elif points.ndim == 2:
        return np.abs((x_end - x_start)*(y_start - points[:, 1]) - (x_start - points[:, 0])*(y_end - y_start)) \
                / np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    else:
        raise UtilityException(f"Points of dimension {points.ndim} are not 1 or 2")

################
# Useful Classes
################

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

####################
# Dataset operations
####################

def create_sample_pattern(sample_pattern):
    """Given a string of '/' separated words, create a dict of the words and their ordering in the string. Idempotent.

    Parameters
    ----------
    sample_pattern : str or (list of str)
        String of '/' separated words

    Returns
    -------
    dict of str: int
        Empty dict if sample pattern is ''.
        Otherwise each key is a word with value that is the index in the patch ID containing the label corresponding to the word.
    """
    if sample_pattern == '':
        return { }
    elif isinstance(sample_pattern, str):
        sample_pattern = sample_pattern.split('/')
        return {k: i for i,k in enumerate(sample_pattern)}
    else:
        return sample_pattern

class IDMaker(object):
    """Create ID string from ID primitives. For example when constructed via
    
    ```
    carla_id_maker = IDMaker(
        'map_name/episode/agent/frame',
        prefixes={
            'episode':  'ep',
            'agent':    'agent',
            'frame':    'frame'},
        format_spec={
            'episode':  '03d',
            'agent':    '03d',
            'frame':    '08d'})
    ```
    
    Then calling
    
    ```
    carla_id_maker.make_id(map_name='Town04', episode=1, agent=123, frame=1000)
    ```
    
    gives

    Town04/ep001/agent123/frame00001000
    """

    @staticmethod
    def __clamp(s):
        return "{" + str(s) + "}"

    def make_fstring(self):
        def f(w):
            s = self.__clamp(f"{w}:{self.__format_spec[w]}")
            return self.__prefixes[w] + s
        l = map(f, self.__sample_pattern_lst)
        return '/'.join(l)

    def __init__(self, s, prefixes={}, format_spec={}):
        self.__sample_pattern_str = s
        self.__sample_pattern_lst = s.split('/')
        self.__sample_pattern = create_sample_pattern(s)
        self.__prefixes = prefixes
        self.__format_spec = format_spec
        for w in self.__sample_pattern_lst:
            if w not in self.__prefixes:
                self.__prefixes[w] = ''
            if w not in self.__format_spec:
                self.__format_spec[w] = ''
        self.__fstring = self.make_fstring()

    @property
    def sample_pattern(self):
        return self.__sample_pattern
    
    @property
    def fstring(self):
        return self.__fstring
        
    def make_id(self, map_name, episode, agent, frame):
        return self.__fstring.format(**locals())

    def filter_ids(self, ids, filter, inclusive=True):
        """
        """
        for word in filter.keys():
            if not isinstance(filter[word], (list, np.ndarray)):
                filter[word] = [filter[word]]
            
        sp = self.__sample_pattern
        id_nd = np.array([[*id.split('/'), id] for id in ids])
        common_words = set(sp) & set(filter)
        b_nd = np.zeros((len(ids), len(common_words)), dtype=bool)
        for idx, word in enumerate(common_words):
            values = filter[word]
            wd_nd = id_nd[:, sp[word]]
            f = lambda v: wd_nd == v 
            wd_nd_b = map_to_ndarray(f, values)
            b_nd[:, idx] = np.any(wd_nd_b, axis=0)
        if inclusive:
            b_nd = np.all(b_nd, axis=1)
        else:
            b_nd = np.any(b_nd, axis=1)
            b_nd = np.logical_not(b_nd)
        id_nd = id_nd[b_nd, -1]
        return id_nd.tolist()
    
    def group_ids(self, ids, words):
        """Group IDs by in the order of the words in the words array.
        For example if sample_pattern of IDs is
        'annotation/subtype/slide/patch_size/magnification'
        and we have IDs like

        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/512/20/2_2
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0

        Setting words=['patch_size', 'magnification'] gives

        512: {
            20: [
                Stroma/MMRd/VOA-1000A/512/20/0_0
                Stroma/MMRd/VOA-1000A/512/20/2_2
            ],
            10: [
                Stroma/MMRd/VOA-1000A/512/10/0_0
            ]
        },
        256: {
            20: [
            ],
            10: [
                Stroma/MMRd/VOA-1000A/256/10/0_0
                Tumor/POLE/VOA-1000B/256/10/0_0
            ]
        }

        Parameters
        ==========
        ids : iterable of str
            List of sample IDs to group.
        words : list of str
            Words to group IDs by. Order of nested labels correspond to order of words array.
        sample_pattern : dict of (str: int)
            Dictionary describing the structure of the patch ID.
            The words for RL experiments can be 'map', 'episode'.
            The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
        
        Returns
        =======
        dict
            The grouped IDs.
            Each group is a list.
            The keys are strings.
        dict of str: list
            Labels corresponding to each word in words array.
            The labels are strings.
        """
        sp = self.__sample_pattern
        id_nd = np.array([[*id.split('/'), id] for id in ids], dtype=np.dtype('U'))
        word_to_labels = { }
        for word in words:
            word_to_labels[word] = np.unique(id_nd[:, sp[word]]).tolist()
        def traverse_words(part_id_nd, idx=0):
            if idx >= len(words):
                return part_id_nd[:, -1].tolist()
            else:
                word = words[idx]
                out = { }
                for label in word_to_labels[word]:
                    selector = part_id_nd[:, sp[word]] == label
                    out[label] = traverse_words(
                            part_id_nd[selector, :],
                            idx=idx + 1)
                return out
        return traverse_words(id_nd), word_to_labels

    def group_ids_by_index(self, ids, include=[], exclude=[]):
        """Group IDs by patch pattern words.
        For example if patch_pattern of IDs is
        'annotation/subtype/slide/patch_size/magnification'
        and we have IDs like

        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/512/20/2_2
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0

        Setting include=['patch_size', 'magnification'] gives

        512/10: [
            Stroma/MMRd/VOA-1000A/512/10/0_0
        ],
        512/20: [
            Stroma/MMRd/VOA-1000A/512/20/0_0
            Stroma/MMRd/VOA-1000A/512/20/2_2
        ],
        256/10: [
            Stroma/MMRd/VOA-1000A/256/20/0_0
        ],
        256/20: [
            Stroma/MMRd/VOA-1000A/256/10/0_0
            Tumor/POLE/VOA-1000B/256/10/0_0
        ]

        Setting exclude=['slide', 'magnification'] gives

        Stroma/MMRd/VOA-1000A/0_0: [
            Stroma/MMRd/VOA-1000A/512/20/0_0
            Stroma/MMRd/VOA-1000A/512/10/0_0
            Stroma/MMRd/VOA-1000A/256/20/0_0
            Stroma/MMRd/VOA-1000A/256/10/0_0
        ],
        Stroma/MMRd/VOA-1000A/2_2: [
            Stroma/MMRd/VOA-1000A/512/20/2_2
        ],
        Tumor/POLE/VOA-1000B/0_0: [
            Tumor/POLE/VOA-1000B/256/10/0_0
        ]

        Parameters
        ----------
        patch_ids : list of str
        patch_pattern : dict
            Dictionary describing the directory structure of the patch paths.
            The words are 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'
        include : iterable of str
            The words to group by. By default includes all words.
        
        exclude : iterable of str
            The words to exclude.
        Returns
        -------
        dict of str: list
            The patch IDs grouped by words.
        """
        sp = self.__sample_pattern
        id_nd = np.array([[*id.split('/'), id] for id in ids], dtype=np.dtype('U'))
        words = set(sp) - set(exclude)
        if include:
            words = words & set(include)
        indices = sorted([sp[word] for word in words] + [id_nd.shape[1] - 1])
        id_nd = id_nd[:,indices]
        id_nd = np.apply_along_axis(lambda r: np.array(['/'.join(r[:-1]), r[-1]]), 1, id_nd)
        group = { }
        for common_id, id in id_nd:
            if common_id not in group:
                group[common_id] = []
            group[common_id].append(id)
        return group

def create_sample_id(path, sample_pattern=None, rootpath=None):
    """Create sample ID from path either by
    1) sample_pattern to find the words to use for ID
    2) rootpath to clip the patch path from the left to form patch ID

    Parameters
    ----------
    path : string
        Absolute path to a patch
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch path.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    rootpath : str
        The root directory path containing sample to clip from sample file path.
        Assumes file path contains rootpath.

    Returns
    -------
    str
        Sample ID generated from path.
    """
    if sample_pattern is not None:
        len_of_patch_id = -(len(sample_pattern) + 1)
        patch_id = strip_extension(path).split('/')[len_of_patch_id:]
        return '/'.join(patch_id)
    elif rootpath is not None:
        return strip_extension(path[len(rootpath):].lstrip('/'))
    else:
        return ValueError("Either sample_pattern or rootpath should be set.")

def create_sample_ids(paths, sample_pattern=None, rootpath=None):
    """Apply create_sample_id() for a list of paths.

    Parameters
    ----------
    path : string
        Absolute path to a patch
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch path.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    rootpath : str
        The root directory path containing sample to clip from sample file path.
        Assumes file path contains rootpath.

    Returns
    -------
    str
        Sample ID generated from path.
    """
    ids = [None]*len(paths)
    for idx, path in enumerate(paths):
        ids[idx] = create_sample_id(path,
                sample_pattern=sample_pattern,
                rootpath=rootpath)
    return ids

def label_from_id(sample_id, word, sample_pattern):
    """Get label corresponding to word from sample ID.

    Parameters
    ----------
    sample_id : str
        Sample ID get label from
    word : str
        Word to the label corresponds to.
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch ID.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    Returns
    -------
    int
        Patch size
    """
    return int(sample_id.split('/')[sample_pattern[word]])    

def index_ids(ids, sample_pattern, include=[], exclude=[]):
    """Index IDs by sample pattern words.
    For example if sample_pattern of IDs is 'annotation/subtype/slide/patch_size/magnification' and we have IDs like

    Stroma/MMRd/VOA-1000A/512/20/0_0
    Stroma/MMRd/VOA-1000A/512/10/0_0
    Stroma/MMRd/VOA-1000A/512/20/2_2
    Stroma/MMRd/VOA-1000A/256/20/0_0
    Stroma/MMRd/VOA-1000A/256/10/0_0
    Tumor/POLE/VOA-1000B/256/10/0_0

    Setting include=['patch_size'] gives

    512/0_0: [
        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
    ],
    512/2_2: [
        Stroma/MMRd/VOA-1000A/512/20/2_2
    ],
    256/0_0: [
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0
    ]

    So here we create meta IDs of form 'patch_size/patch_id' that sample IDs are grouped into.
    Setting exclude=['patch_size', 'magnification'] gives

    Stroma/MMRd/VOA-1000A/0_0: [
        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
    ],
    Stroma/MMRd/VOA-1000A/2_2: [
        Stroma/MMRd/VOA-1000A/512/20/2_2
    ],
    Tumor/POLE/VOA-1000B: [
        Tumor/POLE/VOA-1000B/256/10/0_0
    ]

    Parameters
    ----------
    patch_ids : list of str

    sample_pattern : dict
        Dictionary describing the directory structure of the patch paths.
        The words are 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'

    include : iterable of str
        The words to group by. By default includes all words.
    
    exclude : iterable of str
        The words to exclude.

    Returns
    -------
    dict of str: list
        The patch IDs grouped by words.
    """
    id_nd = np.array([[*id.split('/'), id] for id in ids], dtype=np.dtype('U'))
    words = set(sample_pattern) - set(exclude)
    if include:
        words = words & set(include)
    indices = sorted([sample_pattern[word] for word in words] + [
            id_nd.shape[1] - 2, id_nd.shape[1] - 1])
    id_nd = id_nd[:,indices]
    id_nd = np.apply_along_axis(lambda r: np.array(['/'.join(r[:-1]), r[-1]]),
            1, id_nd)
    group = { }
    for common_id, id in id_nd:
        if common_id not in group:
            group[common_id] = []
        group[common_id].append(id)
    return group

class NumpyEncoder(json.JSONEncoder):
    """The encoding object used to serialize np.ndarrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_datum(datum, directory, filename):
    """Save datum (dict with ndarray values as JSON file).

    Parameters
    ----------
    datum : dict
        The data to save
    directory : str
        The directory name to save the data.
    filename : str
        The file name to name the data.
        If file name contains '/' separated words then create subfolders for them.
        A filename will have the .json suffix added to them if necessary.
    """
    if not os.path.isdir(directory):
        raise UtilityException(f"{directory} does not exist.")
    if filename.startswith('/'):
        raise UtilityException(f"filename {filename} cannot begin with a '/'.")
    """Create subfolders if necessary"""
    filepath = os.path.join(directory, filename)
    os.makedirs(get_dirname_of(filepath), exist_ok=True)
    """Save the file"""
    filepath = filepath if filepath.endswith('.json') else f"{filepath}.json"
    with open(filepath, 'w') as f:
            json.dump(datum, f, cls=NumpyEncoder)
