
from itertools import chain
from hoapp.ast import Alias, Automaton, Comparison, Edge, Int, LogicOp, State
from collections import defaultdict


def counter():
    def add_one():
        add_one.x += 1
        return Int(add_one.x)
    add_one.x = -1
    return add_one


def makeV1(aut: Automaton):
    exprs = defaultdict(counter())
    # Collect things that type to Boolean
    repl = (aut.collect(x) for x in (Comparison, Int, Alias))
    repl = set(chain.from_iterable(repl))
    # Make order deterministic
    repl = sorted(repl, key=lambda x: x.pprint())
    # Give AP numbers to these expressions
    _ = [exprs[x] for x in repl]

    def make_v1_label(node: State | Edge):
        lbl = node.label.replace_by(exprs) if node.label else None
        if node.obligations:
            obls = [o.replace_by(exprs) for o in node.obligations]
            lbl = LogicOp((lbl, *obls, ), "&") if lbl else LogicOp(obls, "&")
        return lbl

    states = []
    for s in aut.states:
        state_lbl = make_v1_label(s)
        edges = []
        for e in s.edges:
            lbl = make_v1_label(e)
            edges.append(Edge(lbl, tuple(), e.target, e.acc_sig))
        states.append(State(state_lbl, tuple(), s.index, s.name, s.acc_sig, edges))

    aps = sorted(exprs.keys(), key=exprs.get)
    aps = [x.pprint() for x in aps]
    return Automaton(
        "v1", aut.name, aut.tool, len(states),
        aut.start, aps, None, None, states, aut.acceptance_sets,
        aut.acceptance, tuple(), tuple(), tuple())
