from collections import defaultdict
from dataclasses import replace
from itertools import chain
from typing import Mapping

from hoapp.ast import (Alias, Automaton, BinaryOp, Edge, Expr, Identifier,
                       InfixOp, Int, State)
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
        return type(e) is BinaryOp and e.op == ":="

    def remove_obligations(e: Expr):
        if isinstance(e, InfixOp):
            ops = tuple(x for x in e.operands if not is_obligation(x))
            return InfixOp(ops, e.op)
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
            edges.append(replace(e, label=lbl, obligations=ob))
        states.append(replace(s, label=state_lbl, obligations=state_ob, edges=tuple(edges)))  # noqa: E501
    return replace(v1pp, states=tuple(states))


def makeV1(aut: Automaton):
    exprs: Mapping[Expr, int] = defaultdict(counter())
    # Collect things that type to Boolean
    collect = (aut.collect(x) for x in (BinaryOp, Int, Alias))
    # Make order deterministic
    repl = sorted(set(chain.from_iterable(collect)), key=lambda x: x.pprint())
    # Give AP numbers to these expressions
    _ = [exprs[x] for x in repl]

    def make_v1_label(node: State | Edge):
        lbl = node.label.replace_by(exprs) if node.label else None
        if node.obligations:
            obls = [o.replace_by(exprs) for o in node.obligations]
            lbl = InfixOp((lbl, *obls, ), "&") if lbl else InfixOp(tuple(obls), "&")  # noqa: E501
        return lbl

    states = []
    for s in aut.states:
        state_lbl = make_v1_label(s)
        edges = []
        for e in s.edges:
            lbl = make_v1_label(e)
            edges.append(replace(e, label=lbl, obligations=()))
        states.append(replace(s, label=state_lbl, obligations=(), edges=tuple(edges)))  # noqa: E501

    aps = (x.pprint() for x in sorted(exprs.keys(), key=lambda x: exprs[x]))

    return replace(
        aut, version=Identifier("v1"), num_states=len(states), ap=tuple(aps),
        controllable_ap=(), aptype=(), states=tuple(states))
