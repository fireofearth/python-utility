from . import grouper

def cube():
    return """
         +--------+
        /        /|
       /  top   / |
      /        /  |
     /        /   |
    +--------+    |
    |        |side|
    |        |    +
    | front  |   /
    |        |  /
    |        | /
    |        |/
    +--------+
    """

def results(*args, start=None, end=None):
    """Print results in documentation"""
    if start is not None:
        print(start)
        print()
    groups = grouper(args, 2)
    for i, (s, o) in enumerate(grouper(args, 2)):
        print(s)
        print(o)
        if i < len(groups) - 1:
            print()
    if end is not None:
        print(end)