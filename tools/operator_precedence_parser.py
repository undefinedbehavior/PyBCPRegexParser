import sys
import re
import pprint
from bcc.tools.utilities import tips

__author__ = 'Yirui Zhang'

class RuleParser:
    def __init__(self):
        self.index = 0
        self.productions = None
        self.tokens = None
        self.production = None

    def _read_(self):
        """ Read a token and return, move index to next one."""

        if self.index >= len(self.tokens):
            return None
        else:
            tmp = self.index
            self.index += 1
            return self.tokens[tmp]

    def _lookahead_(self):
        """Return the current lookahead token."""

        if self.index >= len(self.tokens):
            return None
        else:
            return self.tokens[self.index]

    def _match_(self, str):
        """Match the lookahead token with the string given by user."""

        assert self._read_() == str


    def _parse_head_(self):
        """Parse production header."""

        token = self._read_()
        token = re.sub(r'<|>', '', token)   # remove '<' and '>'
        self.production['head'] = token

    def _parse_tail_(self):
        """Parse one of the production tails"""

        token = self._read_()
        tail = []
        while token != '|' and token is not None:
            if re.match(r'<.+>', token):
                token = re.sub(r'<|>', '', token)
                is_non_t = True
            elif re.match(r'\".+\"', token):
                token = re.sub(r'"', '', token)
                is_non_t = False
            else:
                raise RuntimeError('Wrong token in tail.')
            tail.append((token, is_non_t))
            token = self._read_()
        return tail

    def _parse_tails_(self):
        """Parse production tails"""

        self.production['tails'] = []
        while self._lookahead_() is not None:        # parse rule part
            tail = self._parse_tail_()
            self.production['tails'].append(tail)

    def _parse_rule_(self, rule):
        """ Parse rule"""

        self.index = 0
        self.production = dict()
        self.tokens = re.split(r'[ \t]+', rule)
        self._parse_head_()
        self._match_('::=')
        self._parse_tails_()
        self.productions[self.production['head']] = self.production['tails']

    def _parse_rules_(self, content):
        """Parse rules"""

        rules = re.split(r'[\n\r]+', content)
        for rule in rules:
            if len(rule) is 0:
                continue
            self._parse_rule_(rule)


    def _check_productions_(self):
        """Check whether the production have no errors"""

        # check if there is a non-terminal element without production
        for tails in self.productions.values():
            for tail in tails:
                for (ele, is_non_t) in tail:
                    assert not is_non_t or (is_non_t and (ele in self.productions))


    def parse(self, content):
        """Parse the raw content of rule file"""

        self.productions = dict()
        self.operands = set()
        self._parse_rules_(content)
        self._check_productions_()
        return self.productions


def compute_first_vts(productions):
    """ Compute the first vt set"""

    first_vts = dict((x, set()) for x in productions.keys())
    flag = True
    while flag:
        flag = False
        for head, tails in productions.items():
            for tail in tails:
                if not tail:                          # pass the empty production
                    continue
                new_firsts = []
                if not tail[0][1]:                          # A->a..., a belongs to firstVT(A)
                    new_firsts.append(tail[0][0])
                elif tail[0][0] in first_vts:               # A->B..., firstVT(B) belongs to firstVT(A)
                    new_firsts += first_vts[tail[0][0]]
                    if len(tail) > 1 and not tail[1][1]:    # A->Ba..., a belongs to firstVT(A)
                        new_firsts.append(tail[1][0])
                first_vt = first_vts[head]
                for ele in new_firsts:
                    if ele not in first_vt: # found new first
                        flag = True
                        first_vt.add(ele)
    return first_vts


def compute_last_vts(productions):
    """ Compute last vt set"""

    last_vts = dict((x, set()) for x in productions.keys())
    flag = True
    while flag:
        flag = False
        for head, tails in productions.items():
            for tail in tails:
                if not tail:                                  # pass the empty production
                    continue
                new_firsts = []
                if not tail[-1][1]:                                 # A->...a, a belongs to lastVT(A)
                    new_firsts.append(tail[-1][0])
                elif tail[0][0] in last_vts:                        # A->...B, lastVT(B) belongs to lastVT(A)
                    new_firsts += last_vts[tail[-1][0]]
                    if len(tail) > 1 and not tail[-2][1]:           # A->...aB, a belongs to lastVT(A)
                        new_firsts.append(tail[-2][0])
                last_vt = last_vts[head]
                for ele in new_firsts:
                    if ele not in last_vt:  # found new first
                        flag = True
                        last_vt.add(ele)
    return last_vts


def compute_operator_relations(first_vts, last_vts, result):
    """ Compute the relationships between every two operators. """

    productions = result
    #operators = set(x for s in first_vts.values() for x in s) | set(x for s in last_vts.values() for x in s)
    relations = dict()
    for tails in productions.values():
        for tail in tails:
            for (i, (ele, is_non_t)) in enumerate(tail):
                has_previous = i > 0
                has_next = i < len(tail) - 1
                if not is_non_t:    # is terminal element
                    if has_previous:
                        previous = tail[i - 1][0]
                        previous_is_non_t = tail[i - 1][1]
                        if previous_is_non_t: # previous is a non-terminal element
                            ## for A->...Ba..., add (b in lastVT(B)) > a
                            relations.update({(x, ele): '>' for x in last_vts[previous]})
                    if has_next:
                        next = tail[i + 1][0]
                        next_is_non_t = tail[i + 1][1]
                        if next_is_non_t:   # next is a non-terminal element
                            ## for A->...aB..., add a < (b in firstVT(B))
                            relations.update({(ele, x): '<' for x in first_vts[next]})
                        else:               # next is terminal element
                            ## for A->...ab..., a = b
                            relations[(ele, next)] = '='
                ## for A->...aBb..., a = b
                elif has_next and has_previous and (not tail[i - 1][1]) and (not tail[i + 1][1]):
                    relations[(tail[i - 1][0], tail[i + 1][0])] = '='
    return relations

class CircleInPriorityError (Exception):
    def __init__(self, value = None):
        self.value = value

    def __str__(self):
        return repr(self.value)

def _sub_find_depth_(graph, start, visited):
    if start not in graph:
        return 0
    if start in visited:
        raise CircleInPriorityError()
    t = graph[start]
    visited.add(start)
    maxDepth = 0
    for p in t:
        depth = _sub_find_depth_(graph, p, visited)
        if depth > maxDepth:
            maxDepth = depth
    visited.remove(start)
    return maxDepth + 1

def _find_depth_(graph, start):
    visited = set()
    return _sub_find_depth_(graph, start, visited)

def compute_operator_functions(relations):
    """ Compute the operator functions if there exists one """

    groups = set()
    fMap = dict()
    gMap = dict()
    graph = dict()
    operators = set()
    # construct equals group
    for (a, b), rel in relations.items():
        if rel == '=':
            ft = ('f', a)
            gt = ('g', b)
            group = (ft, gt)
            groups.add(group)
            fMap[a] = group
            gMap[b] = group
            operators.add(a)
            operators.add(b)
        # construct other groups and the graph
    for (a, b), rel in relations.items():
        # ignore the equals relation
        if rel == '=':
            continue
            # add group for f
        if a not in fMap:
            ft = ('f', a)
            group = (ft,)
            groups.add(group)
            fMap[a] = group
            operators.add(a)
            # add group for g
        if b not in gMap:
            gt = ('g', b)
            group = (gt,)
            groups.add(group)
            gMap[b] = group
            operators.add(b)
            # get the groups
        faGroup = fMap[a]
        gbGroup = gMap[b]
        # construct the edge of the graph
        if rel == '<':
            if gbGroup not in graph:
                graph[gbGroup] = set()
            graph[gbGroup].add(faGroup)
        elif rel == '>':
            if faGroup not in graph:
                graph[faGroup] = set()
            graph[faGroup].add(gbGroup)
        # construct the functions
    functions = dict()
    for op in operators:
        # construct f functions
        fResult = None
        if op in fMap:
            group = fMap[op]
            fResult = _find_depth_(graph, group)
            #construct g functions
        gResult = None
        if op in gMap:
            group = gMap[op]
            gResult = _find_depth_(graph, group)
            # merge
        if fResult is None or gResult is None:
            return None
        functions[op] = (fResult, gResult)
    return functions

if __name__ == '__main__':
    sys.stdin = open('../../regex BNF.txt', 'r') # test only, re-direct the standard input
    # deal with the input
    assert len(sys.argv) == 1

    # read content from the rule file
    raw_rule = ''
    for line in sys.stdin:
        raw_rule += line
        # parse rule
    parser = RuleParser()
    result = parser.parse(raw_rule)
    # compute op precedence
    first_vts = compute_first_vts(result)
    last_vts = compute_last_vts(result)
    # outputs IM results
    tips('First VTs:')
    for head, firsts in first_vts.items():
        tips(head, '=', firsts)
    tips('------------')
    tips('Last VTs:')
    for head, lasts in last_vts.items():
        tips(head, '=', lasts)
    tips('------------')
    # compute results
    relations = compute_operator_relations(first_vts, last_vts, result)
    functions = compute_operator_functions(relations)
    # outputs results
    print('Operator Relations:')
    pprint.pprint(relations, width=80, indent=2)
    print('Operator functions:')
    try:
        pprint.pprint(functions, width=80, indent=2)
    except CircleInPriorityError:
        tips('No priority functions exists, circle exists in the dependency graph')
