from collections import deque
from bcc.tools.utilities import debug

__author__ = 'Yirui Zhang'
#################### Automaton Simulation ####################
class Automaton:
    def match(self, string):
        pass


class NFAAutomaton(Automaton):
    def __init__(self, start):
        self.NFAStart = start
        closure = self._closure_({self.NFAStart})
        self.DFAStart = MixState(closure[0], closure[1])
        self.DFAStates = dict({self.DFAStart: self.DFAStart})

    def _single_closure_(self, state, results):
        isMatchState = False
        if isinstance(state, EmptyState):
            isMatchState |= self._single_closure_(state.outPointer.out, results)
            isMatchState |= self._single_closure_(state.outPointer1.out, results)
        else:
            results.add(state)
            isMatchState |= isinstance(state, MatchState)
        return isMatchState

    def _closure_(self, states):
        closure = set()
        isMatchState = False
        for state in states:
            isMatchState |= self._single_closure_(state, closure)
        return frozenset(closure), isMatchState

    def _generate_DFA_state_(self, NFAStates, char):
        states = set(s.outPointer.out for s in NFAStates if s.accept(char))
        closure = self._closure_(states)
        if not closure[0]:
            return None
        DFAState = MixState(closure[0], closure[1])
        if DFAState in self.DFAStates:
            DFAState = self.DFAStates[DFAState]
        return DFAState

    def match(self, string):
        state = self.DFAStart
        for char in string:
            if char not in state.transits:
                next = self._generate_DFA_state_(state.NFAStates, char)
                if next is None:
                    return False
                state.transits[char] = next
            state = state.transits[char]
        return state.isMatchState


class MixState:
    """ This state is for On-The-Fly NFA construction. """

    def __init__(self, NFAStates, isMatchState):
        self.NFAStates = NFAStates
        self.transits = dict()
        self.isMatchState = isMatchState

    def __eq__(self, other):
        return self.NFAStates == other.NFAStates

    def __hash__(self):
        return hash(self.NFAStates)

#################### Builtin NFA State Classes ####################
# According to Thompson's algorithm, each NFA state contains no more than 2 outgoing edge
class OutPointer:
    def __init__(self, out=None):
        self.out = out


class NFAState:
    def __init__(self, s=None):
        self.outPointer = OutPointer(s)

    def accept(self, c):
        pass


class MatchState(NFAState):
    def accept(self, c):
        return False

    def __repr__(self):
        return 'Match'


class CharState(NFAState):
    def __init__(self, c):
        super(CharState, self).__init__()
        self.char = c

    def accept(self, c):
        return c == self.char

    def __repr__(self):
        return self.char


class DotState(NFAState):
    def accept(self, c):
        return c in RegexParser._CHARACTER_

    def __repr__(self):
        return '.'


class EmptyState(NFAState):
    def __init__(self, s1=None, s2=None):
        super(EmptyState, self).__init__(s1)
        self.outPointer1 = OutPointer(s2)

    def accept(self, c):
        return False

    def __repr__(self):
        return 'empty'


class WrapperState(NFAState):
    def __init__(self, otherState):
        super(WrapperState, self).__init__(otherState.outPointer)
        self.innerState = otherState
        if isinstance(otherState, EmptyState):
            self.outPointer1 = otherState.outPointer1


class NFAFragment:
    def __init__(self, start=None, tails=set()):
        self.start = start
        self.tails = tails

    def connect(self, start):
        for outPointer in self.tails:
            outPointer.out = start

    def __repr__(self):
        return repr(self.start)

#################### Regular Expression Parser ####################
class RegexParser:
    """ Regex parser """

    # Regular Expression operator <op_char, op_name> dict.
    _OPERATOR_ = {
        '|': 'OR',
        '.': 'DOT',
        '(': 'LEFT-PAREN',
        ')': 'RIGHT-PAREN',
        '[': 'SET-BEGIN',
        ']': 'SET-END',
        '{': 'REPEAT-BEGIN',
        '}': 'REPEAT-END',
        '*': 'KLEENE',
        '+': 'POS-KLEENE',
        '?': 'QUESTION',
        'concat': 'CAT',
        '\\': 'ESCAPE',
        'EOF': 'EOF',
    }
    # Priority Dict
    _PRIORITY_ = {
        '(': (0, 5),
        ')': (6, 0),
        'EOF': (0, 0),
        'char': (6, 5),
        'concat': (4, 3),
        'follow': (4, 5),
        '|': (2, 1)}
    # None letter set
    _NON_LETTER_ = {chr(x) for x in range(32, 48)} | {chr(x) for x in range(58, 65)} | {chr(x) for x in range(91, 97)} |\
                   {chr(x) for x in range(123, 127)}
    # Letter set
    _LETTER_ = {chr(x) for x in range(ord('a'), ord('z') + 1)} | {chr(x) for x in range(ord('A'), ord('Z') + 1)}
    # Digit set
    _DIGIT_ = {chr(x) for x in range(0, 10)}
    # Character set
    _CHARACTER_ = _LETTER_ | _DIGIT_ | (_NON_LETTER_ - set(_OPERATOR_.keys()))
    # Insert 'concat' operator, first half
    _INSERT_CONCAT_A_ = {')', 'char', 'follow'}
    # Insert 'concat' operator, second half
    _INSERT_CONCAT_B_ = {'(', 'char'}
    ### Follow operator functions ###
    def _op_follow_kleene_plus_(self):
        r = self._pop_opnd_()
        split = EmptyState(r.start)
        r.connect(split)
        self._push_opnd_(NFAFragment(r.start, {split.outPointer1}))

    def _op_follow_appear_(self):
        r = self._pop_opnd_()
        split = EmptyState(r.start)
        r.tails.add(split.outPointer1)
        self._push_opnd_(NFAFragment(split, r.tails))

    def _op_follow_kleene_(self):
        r = self._pop_opnd_()
        split = EmptyState(r.start)
        r.connect(split)
        self._push_opnd_(NFAFragment(split, {split.outPointer1}))

    # Follow Dict
    _FOLLOW_ = {
        '+': _op_follow_kleene_plus_,
        '?': _op_follow_appear_,
        '*': _op_follow_kleene_}
    ### Operator functions ###
    def _op_char_(self, op):
        self._push_opnd_(NFAFragment(op, {op.outPointer}))

    def _op_concat_(self, op):
        r = self._pop_opnd_()
        l = self._pop_opnd_()
        l.connect(r.start)
        self._push_opnd_(NFAFragment(l.start, r.tails))

    def _op_or_(self, op):
        r = self._pop_opnd_()
        l = self._pop_opnd_()
        split = EmptyState(l.start, r.start)
        frag = NFAFragment(split, r.tails | l.tails)
        self._push_opnd_(frag)

    def _op_follow_(self, op):
        assert op in RegexParser._FOLLOW_
        RegexParser._FOLLOW_[op](self)

    _OP_FUNCTIONS_ = {
        'char': _op_char_,
        'concat': _op_concat_,
        '|': _op_or_,
        'follow': _op_follow_
    }
    ### Special operator functions  ###
    def _read_for_plugin_(self):
        self._index_ += 1
        assert self._index_ < len(self._string_)
        return self._string_[self._index_]

    def _next_for_plugin_(self):
        if self._index_ + 1 > len(self._string_):
            return None
        else:
            return self._string_[self._index_ + 1]

    def _match_for_plugin_(self, c):
        char = self._read_for_plugin_()
        assert char == c

    def _plugin_escape_(self):
        char = self._read_for_plugin_()
        assert char in RegexParser._OPERATOR_
        return 'char', CharState(char)

    def _plugin_set_(self):
        debug('processing set')
        regexSet = RegexSet(self)
        return 'char', SetNFAState(regexSet._set_())

    def _plugin_dot_(self):
        return 'char', DotState()

    _PLUGIN_ = {
        '\\': _plugin_escape_,
        '.': _plugin_dot_,
        '[':_plugin_set_
    }

    def __init__(self):
        self._reset_()

    def _reset_(self):
        self._index_ = 0
        self._operands_ = deque()
        self._operators_ = deque()
        self._string_ = None
        self._lookahead_ = None
        self._last_ = deque()
        self._next_ = deque()

    def _pop_opnd_(self):
        if not self._operands_:
            raise RuntimeError('illegal regex: missing operand')
        return self._operands_.pop()

    def _pop_op_(self):
        if not self._operators_:
            raise RuntimeError('illegal regex: missing operator')
        return self._operators_.pop()

    def _push_opnd_(self, opnd):
        self._operands_.append(opnd)

    def _push_op_(self, op):
        self._operators_.append(op)

    def _done_(self):
        return self._index_ > len(self._string_)

    def _tokenize_(self, c):
        if c in RegexParser._FOLLOW_:
            return 'follow', c
        elif c in RegexParser._CHARACTER_:
            return 'char', CharState(c)
        elif c in RegexParser._PLUGIN_:
            return RegexParser._PLUGIN_[c](self)
        else:
            return c, c

    def _rollback_(self):
        if self._lookahead_ is not None:
            self._next_.append(self._lookahead_)
        if self._last_:
            self._lookahead_ = self._last_.pop()
        else:
            self._lookahead_ = None

    def _read_(self):
        if self._lookahead_ is not None:
            self._last_.append(self._lookahead_)
        if self._done_():
            raise RuntimeError('illegal regex: unexpected ending')
        elif self._next_:
            self._lookahead_ = self._next_.pop()
        elif self._index_ == len(self._string_):
            self._lookahead_ = 'EOF', 'EOF'
            self._index_ += 1
        else:
            self._lookahead_ = self._tokenize_(self._string_[self._index_])
            self._index_ += 1
        return self._lookahead_

    def _match_(self, s):
        assert self._read_()[0] == s

    def _peek_op_(self):
        if not self._operators_:
            raise RuntimeError('illegal regex: missing operator')
        return self._operators_[-1]

    def _compare_(self, a, b):
        pTopOp = RegexParser._PRIORITY_[a[0]][0]
        pInputOp = RegexParser._PRIORITY_[b[0]][1]
        debug('cmp [', a[0], ',', b[0], ']=', pTopOp - pInputOp)
        return pTopOp - pInputOp

    def _cleanup_(self):
        matchState = MatchState()
        frag = self._pop_opnd_()
        frag.connect(matchState)
        return frag.start

    def _need_insert_concat_(self):
        return self._last_ and self._last_[-1][0] in RegexParser._INSERT_CONCAT_A_\
        and self._lookahead_[0] in RegexParser._INSERT_CONCAT_B_

    def parse(self, regex):
        self._reset_()
        self._string_ = regex
        self._push_op_(('EOF', 'EOF'))
        while not self._done_():
            self._read_()
            if self._need_insert_concat_():
                self._rollback_()
                self._lookahead_ = 'concat', 'concat'
            cmp = self._compare_(self._peek_op_(), self._lookahead_)
            while cmp > 0:  # Pop until top of stack is less than the input
                op = self._pop_op_()
                debug('pop ', op)
                assert op[0] in RegexParser._OP_FUNCTIONS_
                RegexParser._OP_FUNCTIONS_[op[0]](self, op[1])
                cmp = self._compare_(self._peek_op_(), self._lookahead_)
            if cmp == 0:
                op = self._pop_op_()
                debug('pop ', op)
            else:
                self._push_op_(self._lookahead_)
                debug('push ', self._lookahead_)
        start = self._cleanup_()
        assert not self._operators_
        assert not self._operands_
        return NFAAutomaton(start)

#################### Regular Expression Parser Plugins ####################
class SetNFAState(NFAState):
    def __init__(self, setAst):
        super(SetNFAState, self).__init__()
        self.ast = setAst

    def accept(self, c):
        return self.ast.accept(c)

class SetAstNode:
    def accept(self, c):
        pass

class CharRange(SetAstNode):

    def __init__(self, f = None, t = None):
        self.f = f
        self.t = t
    def accept(self, c):
        return self.f <= ord(c) <= self.t

class OrNode(SetAstNode):
    def __init__(self, asts = None):
        self.asts = asts

    def accept(self, c):
        for ast in self.asts:
            if ast.accept(c):
                return True
        return False

class AndNode(SetAstNode):
    def __init__(self, l, r):
        self.left = l
        self.right = r

    def accept(self, c):
        return self.left.accept(c) and self.right.accept(c)

class NegativeOperator(SetAstNode):
    def __init__(self, ast):
        self.ast = ast

    def accept(self, c):
        return not self.ast.accept(c)

### Set Plugin ###
class RegexSet:

    _ESCAPE_ = {
        '-': 'DASH',
        '\\': 'ESCAPE',
        '[': 'SET-BEGIN',
        ']': 'SET-END',
        '&': 'SINGLE-AND'
    }

    def __init__(self, parser):
        self.parser = parser

    def _item_(self):
        read = self.parser._read_for_plugin_
        char = read()
        if char == '\\':
            char = read()
            assert char in RegexSet._ESCAPE_
        return ord(char)

    def _range_(self):
        nextChar = self.parser._next_for_plugin_
        match = self.parser._match_for_plugin_
        char = nextChar()
        if char == '[':
            match('[')
            return self._set_()

        f = self._item_()
        if nextChar() == '-':
            match('-')
            t = self._item_()
        else:
            t = f
        return CharRange(f, t)

    def _or_(self):
        nextChar = self.parser._next_for_plugin_
        match = self.parser._match_for_plugin_
        char = nextChar()
        stack = []
        while char != ']':
            node = self._range_()
            char = nextChar()
            if char == '&':
                match('&')
                match('&')
                another = self._range_()
                node = AndNode(node, another)
            stack.append(node)
            char = nextChar()
        match(']')
        orNode = OrNode(stack)
        # OR node can not be empty
        assert orNode.asts
        debug(orNode)
        return orNode

    def _set_(self):
        char = self.parser._next_for_plugin_()
        node = None
        if char == '^':
            self.parser._match_for_plugin_('^')
            node = NegativeOperator(self._or_())
        else:
            node = self._or_()
        return node
