
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


def makeV1pp(v1: Automaton, v1pp: Automaton):
    p = parser("expr_or_obligation")
    ap2ast = {Int(i): p.parse(x) for i, x in enumerate(v1.ap)}

    def is_obligation(e: Expr):
        return type(e) is Comparison and e.op == ":="

    def remove_obligations(e: Expr):
        if isinstance(e, LogicOp):
            ops = tuple(x for x in e.operands if not is_obligation(x))
            return LogicOp(ops, e.op)
        return e

    def handle_label(node: State | Edge):
        if node.label is None:
            return None, tuple()
        obligations = [
            ap2ast[x] for x in node.label.collect(Int)
            if is_obligation(ap2ast[x])]
        lbl = node.label.replace_by(ap2ast)
        lbl = remove_obligations(lbl)
        return lbl, tuple(obligations)

    states = []
    for s in v1.states:
        state_lbl, state_ob = handle_label(s)
        edges = []
        for e in s.edges:
            lbl, ob = handle_label(e)
            edges.append(Edge(e.target, e.acc_sig, lbl, ob))
        states.append(State(s.index, s.name, state_lbl, state_ob, s.acc_sig, tuple(edges)))  # noqa: E501
    return Automaton(
        Identifier("v1pp"), v1pp.name, v1pp.tool, v1pp.num_states,
        v1pp.start, v1pp.ap, v1pp.aptype, v1pp.controllable_ap,
        tuple(states), v1pp.acceptance_sets, v1pp.acceptance,
        v1pp.aliases, v1pp.properties, v1pp.headers)


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
