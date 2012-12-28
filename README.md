Yet another front-end compiler-compiler.

ChangeLog
=========
12/28/2012
+ operator precedence parser (operator_precedence_parser.py), which can parse the rule of a operator grammar and
  generate the priority relationship or priority function (if exists one) according to the input rule file.
+ regex parser (re.py), which can accept BRE and part of the ERE (still in process). It will parse the regex to NFA
  and simulate NFA on the fly. Moreover, it will generate DFA state will simulating NFA, and use DFA state instead of
  NFA state later to achieve higher performance. According to dragon book, this implementation will always achieve a
  good balance between DFA and NFA.