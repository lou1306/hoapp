from dataclasses import dataclass
from enum import StrEnum

from lark import Token


class Type(StrEnum):
    BOOL = "bool"
    INT = "int"
    REAL = "auto"


class Expr:
    """Expression abstract class, mainly just for type hinting
    """
    def set_tok(self, tok: Token):
        self.tok = tok

    def pprint(self):
        return str(self)

# Terminals ###################################################################


class IntLit(int, Expr):
    def __repr__(self):
        return f"i{super().__repr__()}"


class RealLit(float, Expr):
    def __repr__(self):
        return f"r{super().__repr__()}"


class Int(int, Expr):
    pass


class String(str):
    pass


class Alias(str, Expr):
    pass


class Boolean(Expr):
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "t" if self else "f"


class Identifier(str, Expr):
    pass


# Expression nodes ############################################################


@dataclass(frozen=True)
class USub(Expr):
    operand: Expr

    def pprint(self):
        return f"-({self.operand.pprint()})"


@dataclass(frozen=True)
class LogicOp(Expr):
    operands: tuple[Expr]
    op: str

    def pprint(self):
        x = f" {self.op} ".join(x.pprint() for x in self.operands)
        return f"({x})"


@dataclass(frozen=True)
class Comparison(Expr):
    left: Expr
    op: str
    right: Expr

    def pprint(self):
        return f"({self.left.pprint()} {self.op} {self.right.pprint()})"


# Acceptance conditions #######################################################

class AccCond:
    pass


@dataclass(frozen=True)
class AccAtom(AccCond):
    inf: bool
    neg: bool
    acc_set: int

    def pprint(self):
        inf_fin = "Inf" if self.inf else "Fin"
        neg = "!" if self.neg else ""
        return f"{inf_fin}({neg}{self.acc_set})"


@dataclass(frozen=True)
class AccCompound(AccCond):
    left: AccCond
    op: str
    right: AccCond

    def pprint(self):
        return f"({self.left.pprint()} {self.op} {self.right.pprint()})"

# Automaton nodes #############################################################


@dataclass(frozen=True)
class Edge:
    label: Expr | None
    obligations: tuple[Comparison]
    target: Expr
    acc_sig: any

    def pprint(self):
        ob = ", ".join(x.pprint() for x in self.obligations)
        ob = f" $ {ob}" if ob else ""
        label = "" if self.label is None else f"[{self.label.pprint()}{ob}] "
        sig = "" if self.acc_sig is None else f" {{{' '.join(self.acc_sig)}}}"
        return f"{label}{self.target}{sig}"


@dataclass(frozen=True)
class State:
    label: Expr | None
    index: int
    name: str | None
    acc_sig: tuple
    edges: tuple[Edge]

    def pprint(self):
        label = f"[{self.label.pprint()}] " if self.label is not None else ""
        sig = "" if self.acc_sig is None else f" {{{' '.join(str(x) for x in self.acc_sig)}}}"  # noqa: E501
        return "\n".join((
            f"State: {label}{self.index}{sig}",
            *(e.pprint() for e in self.edges)))


@dataclass(frozen=True)
class Automaton:
    version: Identifier
    name: str | None
    tool: str | tuple[str, str] | None
    num_states: int | None
    start: tuple[Int]
    ap: tuple[str]
    aptype: tuple[Type] | None
    controllable_ap: tuple[Int] | None
    states: tuple[State]
    acceptance_sets: int
    acceptance: AccCond
    aliases: tuple[tuple[str, Expr]]
    properties: tuple[str]
    headers: tuple[tuple[str, any]]

    def pprint(self):
        start = (f"Start: {x}" for x in self.start)
        aliases = (f"Alias: {x[0]} {x[1]}" for x in self.aliases)
        headers = (f"{h}: {v}" for h, v in self.headers)
        controllable = (
            f"""controllable-AP: {" ".join(str(x) for x in self.controllable_ap)}"""
            if self.controllable_ap else "")
        header = (
            f"HOA: {self.version}",
            f"name: {self.name}" if self.name else "",
            f"tool: {self.tool}" if self.tool else "",
            f"States: {self.num_states}" if self.num_states is not None else "",  # noqa: E501
            f"""AP: {len(self.ap)} {" ".join(f'"{x}"' for x in self.ap)}""",
            f"""Acceptance: {self.acceptance_sets} {" ".join(x.pprint() for x in self.acceptance)}""",  # noqa: E501
            controllable, *start, *aliases, *headers
        )
        return "".join((
            "\n".join(x for x in header if x),
            "\n--BODY--\n",
            "\n".join(s.pprint() for s in self.states),
            "\n--END--"))
