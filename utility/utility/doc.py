from . import grouper

def results(*args):
    """Print results in documentation"""
    groups = grouper(args, 2)
    for i, (s, o) in enumerate(grouper(args, 2)):
        print(s)
        print(o)
        if i < len(groups) - 1:
            print()