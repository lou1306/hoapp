from collections import defaultdict
from dataclasses import replace
from itertools import chain
from typing import Mapping, Optional

from hoapp.ast.ast import (Alias, Automaton, BinaryOp, Boolean, Edge, Expr,
                           Identifier, InfixOp, Int, State, String, Type)
from hoapp.parser import mk_parser


def makeV1pp(v1: Automaton, types: Optional[dict[str, Type]] = None) -> Automaton:  # noqa: E501
    p = mk_parser("expr_or_obligation")
    ap2ast = {Int(i): p.parse(x) for i, x in enumerate(v1.ap)}

    def is_obligation(e: Expr):
        return type(e) is BinaryOp and e.op == ":="

    def remove_obligations(e: Expr) -> Expr | None:
        if is_obligation(e):
            return None
        if isinstance(e, InfixOp):
            ops = tuple(x for x in e.operands if not is_obligation(x))
            if len(ops) == 0:
                return Boolean(e.op == "&", None)
            return InfixOp(ops, e.op)
        return e

    def handle_label(node: State | Edge):
        if node.label is None:
            return None, tuple()
        obligations = [
            ap2ast[x] for x in node.label.collect(Int)
            if is_obligation(ap2ast[x])]
        lbl = remove_obligations(node.label.replace_by(ap2ast))
        return lbl, tuple(obligations)

    states = []
    for s in v1.states:
        state_lbl, state_ob = handle_label(s)
        edges = []
        for e in s.edges:
            lbl, ob = handle_label(e)
            edges.append(replace(e, label=lbl, obligations=ob))
        states.append(replace(s, label=state_lbl, obligations=state_ob, edges=tuple(edges)))  # noqa: E501

    v1pp_ap = next((h for h in v1.headers if h[0] == "v1pp-AP"), None)
    if v1pp_ap is not None:
        aps = tuple(v1pp_ap[1][1:])
    else:
        aps_list = []
        p = mk_parser("aname")
        for x in v1.ap:
            try:
                p.parse(x)
                aps_list.append(String(x.replace("@", "")))
            except Exception:
                continue
        aps = tuple(aps_list)

    ap_types: tuple[Type, ...] = ()
    if types:
        ap_types = tuple(types.get(x, Type.BOOL) for x in aps)
    else:
        header = next((h for h in v1.headers if h[0] == "v1pp-AP-type"), None)
        ap_types = () if header is None else header[1]

    aliases = tuple((f"@{ap}", Int(i)) for i, ap in enumerate(aps))
    headers = tuple(h for h in v1.headers if h[0] not in ("v1pp-AP", "v1pp-AP-type"))  # noqa: E501

    return replace(v1, version=Identifier("v1pp"), ap=aps, aliases=aliases,
                   aptype=ap_types,
                   states=tuple(states), headers=headers)


def makeV1(aut: Automaton) -> Automaton:
    """Lower a HOApp automaton into HOAv1.

    Args:
        aut (Automaton): An automaton in HOApp format

    Returns:
        Automaton: A copy of aut, lowered into HOAv1 format.
    """

    def counter():
        def add_one():
            add_one.x += 1
            return Int(add_one.x)
        add_one.x = -1
        return add_one

    aut = aut.auto_alias()
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
    v1pp_ap = ("v1pp-AP", tuple([str(len(aut.ap)), *aut.ap]))

    headers = [v1pp_ap]
    if aut.aptype:
        headers.append(("v1pp-AP-type", tuple(x.value for x in aut.aptype)))

    return replace(
        aut, version=Identifier("v1"), num_states=len(states), ap=tuple(aps),
        states=tuple(states), aliases=(), aptype=(), headers=tuple(headers))
