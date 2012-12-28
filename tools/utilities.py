import sys
import pprint
__author__ = 'Yirui Zhang'

debugFlag = False

def tips(*args, sep=' ', end='\n'):
    print(*args, sep=sep, end=end, file=sys.stderr)

def ptips(object, stream=None, indent=1, width=80, depth=None):
    pprint.pprint(object, stream=stream, indent=indent, width=width, depth=depth)

def emit(*args, sep=' ', end='\n'):
    print(*args, sep=sep, end=end, file=sys.stdout)

def debug(*args, sep=' ', end='\n'):
    if debugFlag:
        print(*args, sep=sep, end=end, file=sys.stderr)