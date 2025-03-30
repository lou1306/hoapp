
from collections import defaultdict
from itertools import chain
from typing import Mapping

from hoapp.ast import (Alias, Automaton, Comparison, Edge, Expr, Identifier,
                       Int, LogicOp, State)
from hoapp.parser import parser


def counter():
    def add_one():
        add_one.x += 1
        return Int(add_one.x)
    add_one.x = -1
    return add_one


def makeV1(aut: Automaton):
    exprs: Mapping[Expr, int] = defaultdict(counter())
    # Collect things that type to Boolean
    collect = (aut.collect(x) for x in (Comparison, Int, Alias))
    # Make order deterministic
    repl = sorted(set(chain.from_iterable(collect)), key=lambda x: x.pprint())
    # Give AP numbers to these expressions
    _ = [exprs[x] for x in repl]

    def make_v1_label(node: State | Edge):
        lbl = node.label.replace_by(exprs) if node.label else None
        if node.obligations:
            obls = [o.replace_by(exprs) for o in node.obligations]
            lbl = LogicOp((lbl, *obls, ), "&") if lbl else LogicOp(tuple(obls), "&")  # noqa: E501
        return lbl

    states = []
    for s in aut.states:
        state_lbl = make_v1_label(s)
        edges = []
        for e in s.edges:
            lbl = make_v1_label(e)
            edges.append(Edge(e.target, label=lbl, acc_sig=e.acc_sig))
        s1 = State(s.index, s.name, state_lbl, acc_sig=s.acc_sig, edges=tuple(edges))  # noqa: E501
        states.append(s1)  # noqa: E501

    aps = tuple(x.pprint() for x in sorted(exprs.keys(), key=lambda x: exprs[x]))  # noqa: E501
    return Automaton(
        Identifier("v1"), aut.name, aut.tool, len(states),
        aut.start, aps, None, tuple(), tuple(states), aut.acceptance_sets,
        aut.acceptance, tuple(), tuple(), tuple())
