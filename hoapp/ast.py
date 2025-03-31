from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Optional


from lark import Token


class Type(Enum):
    BOOL = "bool"
    INT = "int"
    REAL = "auto"

    def __le__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        if self == Type.INT:
            return other in (Type.REAL, Type.INT)
        return self == other


class Expr:
    """Expression abstract class, mainly just for type hinting
    """
    def set_tok(self, tok: Token) -> None:
        self.tok = tok

    def pprint(self, *_) -> str:
        return str(self)

    def collect(self, t) -> Generator:
        if isinstance(self, t):
            yield self

    def replace_by(self, mapping) -> "Expr":
        return mapping.get(self, self)

    def type_check(self, _) -> Type:
        raise NotImplementedError()

# Terminals ###################################################################


class IntLit(int, Expr):
    def __repr__(self):
        return f"i{super().__repr__()}"

    def type_check(self, _) -> Type:
        return Type.INT


class RealLit(float, Expr):
    def __repr__(self):
        return f"r{super().__repr__()}"

    def type_check(self, _) -> Type:
        return Type.REAL


class Int(int, Expr):
    def type_check(self, aut: "Automaton") -> Type:
        if aut.aptype is None:
            return Type.BOOL
        try:
            return aut.aptype[int(self)]
        except KeyError:
            raise TypeError(f"Unknown AP {self}")


class String(str):
    pass


class Alias(str, Expr):
    def type_check(self, aut: "Automaton") -> Type:
        try:
            alias_def = next(x[1] for x in aut.aliases if x[0] == self)
            return alias_def.type_check(aut)
        except StopIteration:
            raise TypeError(f"Undefined alias {self}")


class Boolean(Expr):
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "t" if self else "f"

    def type_check(self, _) -> Type:
        return Type.BOOL


class Identifier(str, Expr):
    pass


# Expression nodes ############################################################


@dataclass(frozen=True)
class USub(Expr):
    operand: Expr

    def pprint(self, paren=False):
        result = f"-{self.operand.pprint(True)}"
        return f"({result})" if paren else result

    def collect(self, t):
        if isinstance(self, t):
            yield self
        yield from self.operand.collect(t)

    def type_check(self, aut: "Automaton") -> Type:
        op_type = self.operand.type_check(aut)
        if not op_type <= Type.REAL:
            raise TypeError(f"Unexpected {op_type} operand in {self}")
        return op_type


@dataclass(frozen=True)
class InfixOp(Expr):
    operands: tuple[Expr, ...]
    op: str

    def __post_init__(self):
        if type(self.operands) is not tuple:
            raise Exception(f"expected tuple, got {type(self.operands)}")

    def pprint(self, paren=False):
        if self.op == "!":
            return f"!{self.operands[0].pprint(True)}"
        x = f" {self.op} ".join(
            x.pprint(len(self.operands) > 1)
            for x in self.operands)
        return f"({x})" if paren else x

    def collect(self, t):
        if isinstance(self, t):
            yield self
        for o in self.operands:
            yield from o.collect(t)

    def replace_by(self, mapping):
        if self in mapping:
            return mapping[self]
        ops = tuple(o.replace_by(mapping) for o in self.operands)
        return InfixOp(operands=ops, op=self.op)

    def type_check(self, aut: "Automaton") -> Type:
        types = [(o, o.type_check(aut)) for o in self.operands]
        if self.op in "&|":
            wrong = [(o, t) for o, t in types if not t <= Type.BOOL]
            result = Type.BOOL
        elif self.op == "*":
            wrong = [(o, t) for o, t in types if not t <= Type.REAL]
            result = (Type.INT if all(t <= Type.INT for _, t in types) else Type.REAL)  # noqa: E501
        else:
            raise TypeError(f"Unexpected operator: {self.op}")
        if wrong:
            raise TypeError(f"Invalid operands for {self.op}: {wrong}")
        return result


@dataclass(frozen=True)
class BinaryOp(Expr):
    left: Expr
    op: str
    right: Expr

    def pprint(self, paren=False):
        result = f"{self.left.pprint(True)} {self.op} {self.right.pprint(self.op != ":=")}"  # noqa: E501
        return f"({result})" if paren else result

    def collect(self, t):
        if isinstance(self, t):
            yield self
        yield from self.left.collect(t)
        yield from self.right.collect(t)

    def replace_by(self, mapping):
        return mapping.get(self, self)

    def type_check(self, aut: "Automaton") -> Type:
        tl, tr = self.left.type_check(aut), self.right.type_check(aut)
        if self.op in "+-":
            error = tl <= Type.BOOL or tr <= Type.BOOL
            result = (Type.INT if all(t <= Type.INT for t in (tl, tr)) else Type.REAL)  # noqa: E501
        else:
            error = tl.subtype_of(Type.REAL) or tr.subtype_of(Type.REAL)
            result = Type.BOOL
        if error:
            raise TypeError(f"Invalid operands for {self.op}: {self}")
        return result

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
    target: Expr
    acc_sig: tuple[int, ...] = field(default_factory=tuple)
    label: Optional[Expr] = None
    obligations: tuple[BinaryOp, ...] = field(default_factory=tuple)

    def pprint(self):
        ob = ", ".join(x.pprint() for x in self.obligations)
        ob = f" $ {ob}" if ob else ""
        label = f"[{self.label.pprint()}{ob}] " if self.label else ""
        sig = f" {{{' '.join(self.acc_sig)}}}" if self.acc_sig else ""
        return f"{label}{self.target}{sig}"

    def collect(self, t):
        if self.label:
            yield from self.label.collect(t)
        for o in self.obligations:
            yield from o.collect(t)


@dataclass(frozen=True)
class State:
    index: int
    name: Optional[str] = None
    label: Optional[Expr] = None
    obligations: tuple[BinaryOp, ...] = field(default_factory=tuple)
    acc_sig: tuple[int, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def pprint(self):
        label = f"[{self.label.pprint()}] " if self.label is not None else ""
        sig = f" {{{' '.join(str(x) for x in self.acc_sig)}}}" if self.acc_sig else ""  # noqa: E501
        return "\n".join((
            f"State: {label}{self.index}{sig}",
            *(e.pprint() for e in self.edges)))

    def collect(self, t):
        if self.label:
            yield from self.label.collect(t)
        for e in self.edges:
            yield from e.collect(t)


@dataclass(frozen=True)
class Automaton:
    version: Identifier
    name: str | None
    tool: str | tuple[str, str] | None
    num_states: int | None
    start: tuple[Int, ...]
    ap: tuple[str, ...]
    aptype: tuple[Type] | None
    controllable_ap: tuple[Int, ...]
    states: tuple[State, ...]
    acceptance_sets: int
    acceptance: AccCond
    aliases: tuple[tuple[str, Expr], ...]
    properties: tuple[str, ...]
    headers: tuple[tuple[str, Any], ...]

    def pprint(self):
        start = (f"Start: {x}" for x in self.start)
        aliases = (f"Alias: {x[0]} {x[1]}" for x in self.aliases)
        headers = (f"{h}: {v}" for h, v in self.headers)
        controllable = (
            f"""controllable-AP: {" ".join(str(x) for x in self.controllable_ap)}"""  # noqa: E501
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

    def collect(self, t):
        for s in self.states:
            yield from s.collect(t)
