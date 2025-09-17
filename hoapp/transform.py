from collections import defaultdict
from dataclasses import replace
from itertools import chain
from typing import Mapping, Optional

from hoapp.ast.automata import Automaton, Label
from hoapp.ast.expressions import (Alias, BinaryOp, Boolean, Expr, Identifier,
                                   InfixOp, Int, String, Type, USub)
from hoapp.parser import mk_parser

CMP_OPS = ("==", "!=", ">=", ">", "<=", "<")


def quote_exprs(expr: Expr):
    match expr:
        case Alias():
            return String(expr.pprint())
        case BinaryOp(op=op) if op in CMP_OPS:
            return String(expr.pprint())
        case BinaryOp(left=left, op=op, right=right):
            lhs, rhs = quote_exprs(left), quote_exprs(right)
            return BinaryOp(left=lhs, op=op, right=rhs)
        case InfixOp(operands=ops, op=op):
            recurse = (quote_exprs(o) for o in ops)
            return InfixOp(operands=tuple(recurse), op=op)
        case USub(operand=o):
            return USub(operand=quote_exprs(o))
        case _:
            return expr


def makeV1pp(v1: Automaton, types: Optional[dict[str, Type]] = None) -> Automaton:  # noqa: E501
    """Turn a (lowered) HOA automaton into HOApp.

    Args:
        v1 (Automaton): A HOA automaton.

        types (Optional[dict[str, Type]], optional): A mapping from AP names \
            to their data types.
            If None is given, the procedure looks for a `v1pp-AP-type` header.
            Lacking that, all APs are assumed to be of Boolean type. \
            Defaults to None.

    Returns:
        Automaton: A HOApp version of `v1`.
    """
    p = mk_parser("expr_or_obligation")
    ap2ast = {Int(i): p.parse(x) for i, x in enumerate(v1.ap)}

    def is_obligation(e: Expr):
        return type(e) is BinaryOp and e.op == ":="

    def remove_obligations(e: Expr) -> Expr | None:
        if is_obligation(e):
            return Boolean(True, None)
        if isinstance(e, InfixOp):
            ops = tuple(remove_obligations(x) for x in e.operands)
            # if len(ops) == 0:
            #     return Boolean(e.op in "&!", None)
            return InfixOp(tuple(o for o in ops if o), e.op)
        return e

    def handle_label(label: Label | None) -> Label | None:
        if label is None or label.guard is None:
            return None
        obligations = [
            ap2ast[x] for x in label.guard.collect(Int)
            if is_obligation(ap2ast[x])]
        lbl = remove_obligations(label.guard.replace_by(ap2ast))
        return Label(guard=lbl, obligations=tuple(obligations))

    states = []
    for s in v1.states:
        state_lbl = handle_label(s.label)
        edges = []
        for e in s.edges:
            lbl = handle_label(e.label)
            edges.append(replace(e, label=lbl))
        states.append(replace(s, label=state_lbl, edges=tuple(edges)))  # noqa: E501

    v1pp_ap = next(iter(v1.get("v1pp-AP", ())), None)
    if v1pp_ap is not None:
        aps = v1pp_ap
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
        aps = tuple(types.keys())
        ap_types = tuple(types.get(x, Type.BOOL) for x in aps)
    else:
        types_header = next(iter(v1.get("v1pp-AP-type")), ())
        ap_types = tuple(Type(t) for t in types_header)

    aliases = tuple((f"@{ap}", Int(i)) for i, ap in enumerate(aps))
    headers = tuple(
        (h[0].replace("v1pp-", ""), h[1])
        for h in v1.headers
        if h[0] not in ("v1pp-AP", "v1pp-AP-type"))  # noqa: E501

    return replace(v1, version=Identifier("v1pp"), ap=aps, aliases=aliases,
                   aptype=ap_types, states=tuple(states), headers=headers)


def makeV1(aut: Automaton) -> Automaton:
    """Lower a HOApp automaton into HOAv1.

    Args:
        aut (Automaton): An automaton in HOApp format.

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

    states = []
    for s in aut.states:
        state_lbl = s.label.make_v1(exprs) if s.label else None
        edges = []
        for e in s.edges:
            lbl = e.label.make_v1(exprs) if e.label else None
            edges.append(replace(e, label=lbl) if lbl else e)
        states.append(replace(s, label=state_lbl, edges=tuple(edges)))  # noqa: E501

    aps = (x.pprint() for x in sorted(exprs.keys(), key=lambda x: exprs[x]))
    v1pp_ap = ("v1pp-AP", tuple([str(len(aut.ap)), *aut.ap]))

    headers = [v1pp_ap]
    if aut.aptype:
        headers.append(("v1pp-AP-type", tuple(x.value for x in aut.aptype)))
    for header in ("assume", "guarantee"):
        for lst in aut.get(header, ()):
            for ltl in lst:
                headers.append((f"v1pp-{header}", (f'"{ltl.pprint()}"', )))
    if aut.controllable_ap:
        str_control_ap = tuple(str(x) for x in aut.controllable_ap)
        headers.append(("v1pp-controllable-AP", str_control_ap))

    return replace(
        aut, version=Identifier("v1"), num_states=len(states), ap=tuple(aps),
        states=tuple(states), aliases=(), aptype=(), controllable_ap=(),
        headers=tuple(headers))
