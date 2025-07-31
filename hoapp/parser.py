from importlib import resources
from pathlib import Path
from typing import Counter, TextIO

from lark import Lark, Token, Transformer

from hoapp.ast.acceptance import AccAtom, AccCompound
from hoapp.ast.automata import Automaton, Edge, Label, State
from hoapp.ast.expressions import (Alias, BinaryOp, Boolean, Identifier,
                                   InfixOp, Int, IntLit, RealLit, String, Type,
                                   USub)

grammar_file = resources.files().joinpath("hoapp.lark")


class MakeAst(Transformer):
    """Main transformer to turn Lark's tree into AST objects."""

    @staticmethod
    def _id(tree):
        return tree

    @staticmethod
    def _head(tree):
        return tree[0]

    @staticmethod
    def _terminal(typ, value, tok=None):
        return typ(value, tok=tok)

    def ANAME(self, tok: Token):
        return MakeAst._terminal(Alias, tok)

    def BOOLEAN(self, tok: Token):
        return MakeAst._terminal(Boolean, tok == "t", tok)

    def INT(self, tok: Token):
        return MakeAst._terminal(Int, tok)

    def INTLIT(self, tok: Token):
        return MakeAst._terminal(IntLit, tok[1:], tok)

    def IDENTIFIER(self, tok):
        return MakeAst._terminal(Identifier, tok)

    def REALLIT(self, tok: Token):
        return MakeAst._terminal(RealLit, tok[1:], tok)

    def STRING(self, tok):
        return MakeAst._terminal(String, tok[1:-1], tok)

    def TYPE(self, tok):
        return Type(tok)

    format_version = _head
    acc_sig = _head
    label = _id
    header = _id
    body = _id
    acc_sig = _id
    state_name = _id

    def header_item(self, tree):
        return {str(tree[0])[:-1]: tree[1] if len(tree) == 2 else tree[1:]}

    def compare(self, tree):
        return BinaryOp(tree[0], tree[1].value, tree[2])

    def addsub(self, tree):
        lhs, op, rhs, *tree = tree
        node = BinaryOp(lhs, op.value, rhs)
        while tree:
            op, rhs, *tree = tree
            node = BinaryOp(node, op.value, rhs)
        return node

    def ltl_fgx(self, tree):
        return InfixOp(tuple(tree[1:]), tree[0])

    def mul(self, tree):
        return InfixOp(tuple(tree), "*")

    def eq(self, tree):
        return self.compare(tree)

    def impl(self, tree):
        return BinaryOp(*tree)

    def until(self, tree):
        return BinaryOp(tree[0], "U", tree[1])

    def conj(self, tree):
        return InfixOp(tuple(tree), "&")

    def disj(self, tree):
        return InfixOp(tuple(tree), "|")

    def state_conj(self, tree):
        return self.conj(tuple(tree))

    def neg(self, tree):
        return InfixOp(tuple(tree), "!")

    def minus(self, tree):
        return USub(tree[0])

    def acceptance_atom(self, tree):
        return AccAtom(tree[0] == "Inf", tree[1] is not None, tree[2])

    def acceptance_conj(self, tree):
        return AccCompound(tree[0], "&", tree[1])

    def acceptance_disj(self, tree):
        return AccCompound(tree[0], "|", tree[1])

    def obligation(self, tree):
        return BinaryOp(tree[0], ":=", tree[1])

    def _handle_label(self, tree) -> Label | None:
        if tree[0] is not None:
            guard, *obligations = tree[0]
            if obligations and obligations[0] is None:
                obligations = ()
            return Label(guard=guard, obligations=tuple(obligations))
        else:
            return None

    def edge(self, tree):
        lbl = self._handle_label(tree)
        target, acc_sig = tree[1:]
        e = Edge(target, acc_sig, lbl)
        return e

    def state(self, tree):
        lbl = self._handle_label(tree[0])
        _, index, name, acc_sig = tree[0]
        edges = tuple(tree[1:])
        return State(int(index), name, lbl, acc_sig, edges)

    def automaton(self, tree):
        canonical_headers = (
            "Acceptance", "acc-name", "Alias",
            "AP", "AP-type", "controllable-AP", "name",
            "properties", "Start", "States", "tool")
        dicts = [x for x in tree[0] if isinstance(x, dict)]
        all_headers = Counter(x for d in dicts for x in d.keys())
        multiple_headers = (
            x for x in all_headers if all_headers[x] > 1
            and x not in
            ("Start", "Alias", "properties", "assume", "guarantee"))
        err_msg = "\n".join(
            f"Too many '{x}:' headers." for x in multiple_headers)
        if err_msg:
            raise Exception(err_msg)

        version = tree[0][0]
        num_states, aps, types, name, tool, num_acc, acc, ctrl_aps = [None] * 8
        start, aliases, properties, headers = [], [], [], []
        for d in dicts:
            num_states = num_states or d.get("States")
            name = name or d.get("name")
            tool = tool or d.get("tool")
            new_start = d.get("Start")
            if new_start is not None:
                start.append(new_start)
            new_ap = d.get("AP")
            if new_ap == 0:
                num_aps, aps = 0, ()
            elif new_ap is not None:
                num_aps, *aps = new_ap
                if len(aps) != num_aps:
                    raise Exception(f"Wrong number of APs (expected {num_aps}, got {len(aps)})")  # noqa: E501
                counter_aps = Counter(aps)
                multiple_aps = [x for x in counter_aps if counter_aps[x] > 1]
                if multiple_aps:
                    raise Exception(f"Duplicate APs: {multiple_aps}")
            new_acc = d.get("Acceptance")
            if new_acc:
                num_acc, *acc = new_acc

            new_alias = d.get("Alias")
            if new_alias is not None:
                aliases.append((new_alias[0], new_alias[1]))
            pp = d.get("properties", [])
            if isinstance(pp, str):
                properties.append(pp)
            else:
                properties.extend(pp)
            ctrl_aps = ctrl_aps or d.get("controllable-AP")
            if isinstance(ctrl_aps, int):
                ctrl_aps = [ctrl_aps]
            types = types or d.get("AP-type")
            types = [Type(t) for t in types or ()]
            others = (k for k in d if k not in canonical_headers)
            for k in others:
                value = d[k] if isinstance(d, (list, tuple)) else (d[k], )
                headers.append((k, value))

        tool = tool if isinstance(tool, str) else tuple(tool) if tool else None
        states = sorted(tree[1], key=lambda x: x.index)

        return Automaton(
            version=version, name=name, tool=tool,
            num_states=num_states, start=tuple(start), states=tuple(states),
            ap=aps, aptype=tuple(types or ()),
            controllable_ap=tuple(ctrl_aps or ()),
            acceptance_sets=num_acc, acceptance=acc[0],
            aliases=tuple(aliases),
            properties=tuple(properties),
            headers=tuple(headers))


def mk_parser(start="test_terminals"):
    with open(grammar_file) as grammar:
        parser = Lark(grammar, start=start, parser="lalr", transformer=MakeAst())  # noqa: E501
    return parser


__HOAPP_PARSER = mk_parser("automaton")


def parse_string(s: str) -> Automaton:
    aut = __HOAPP_PARSER.parse(s)
    return aut


def parse_stream(stream: TextIO) -> Automaton:
    aut = __HOAPP_PARSER.parse(stream.read())
    return aut


def parse(filename: Path) -> Automaton:
    """Parse a .hoa file into an Automaton object.

    Also performs type checking.

    Args:
        filename (Path): Path to a .hoa file

    Returns:
        Automaton: The automaton contained in `filename`.
    """
    with open(filename) as hoa_file:
        return parse_stream(hoa_file)
