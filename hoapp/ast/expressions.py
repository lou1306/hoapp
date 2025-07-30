from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generator

import pysmt.shortcuts as smt  # type: ignore

from .ast import Token, Type

if TYPE_CHECKING:
    from .automata import Automaton

from operator import eq, ge, gt, le, lt, ne, sub

import z3  # type: ignore


class Expr:
    """Expression abstract class, mainly just for type hinting
    """
    def pprint(self, *_) -> str:
        return str(self)

    def collect(self, t) -> Generator:
        if isinstance(self, t):
            yield self

    def replace_by(self, mapping) -> "Expr":
        return mapping.get(self, self)

    def type_check(self, _) -> Type:
        raise NotImplementedError()

    def unalias(self, _) -> "Expr":
        return self

    def auto_alias(self, _) -> "Expr":
        return self


class IntLit(Token, int, Expr):
    def __repr__(self):
        return f"i{super().__repr__()}"

    def type_check(self, _) -> Type:
        return Type.INT


class RealLit(float, Expr):
    def __repr__(self):
        return f"r{super().__repr__()}"

    def type_check(self, _) -> Type:
        return Type.REAL


class String(Token, str):
    def pprint(self, *_) -> str:
        return f'"{self}"'


class Alias(String, Expr):
    def type_check(self, aut: "Automaton") -> Type:
        alias_def = aut.get_alias(self)
        return alias_def.type_check(aut)

    def unalias(self, aut: "Automaton") -> Expr:
        return aut.get_alias(self)

    def pprint(self, *_) -> str:
        return str(self)


class Int(Token, int, Expr):
    def type_check(self, aut: "Automaton") -> Type:
        return aut.get_type(int(self))

    def auto_alias(self, aut: "Automaton") -> Alias:
        try:
            return Alias(f"@{aut.ap[int(self)]}")
        except KeyError:
            raise Exception(f"Invalid ap {self}")


class Boolean(Expr):
    def __init__(self, value, tok):
        self.value = value
        self.tok = tok

    def __bool__(self):
        return self.value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "t" if self else "f"

    def type_check(self, _) -> Type:
        return Type.BOOL


class Identifier(String, Expr):
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

    def unalias(self, aut: "Automaton") -> "USub":
        return replace(self, operand=self.operand.unalias(aut))

    def auto_alias(self, aut: "Automaton") -> "USub":
        return replace(self, operand=self.operand.auto_alias(aut))


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
        if self.op in "&|!":
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

    def unalias(self, aut: "Automaton") -> "InfixOp":
        ops = tuple(o.unalias(aut) for o in self.operands)
        return replace(self, operands=ops)

    def auto_alias(self, aut: "Automaton") -> "InfixOp":
        ops = tuple(o.auto_alias(aut) for o in self.operands)
        return replace(self, operands=ops)


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
            error = not (tl <= Type.REAL and tr <= Type.REAL)
            result = Type.BOOL
        if error:
            raise TypeError(f"Invalid operands for {self.pprint()}: {tl}, {tr}")  # noqa: E501
        return result

    def unalias(self, aut: "Automaton") -> "BinaryOp":
        lhs, rhs = self.left.unalias(aut), self.right.unalias(aut)
        return replace(self, left=lhs, right=rhs)

    def auto_alias(self, aut: "Automaton") -> "BinaryOp":
        lhs, rhs = self.left.auto_alias(aut), self.right.auto_alias(aut)
        return replace(self, left=lhs, right=rhs)

    def compatible_with(self, other: "BinaryOp") -> bool:
        if self.left != other.left:
            return True
        if self.right == other.right:
            return True
        return False


Z3_OPS = {
    "&": z3.And, "|": z3.Or, "+": z3.Sum, "-": sub, "*": z3.Product,
    "==": eq, "!=": ne, "<": lt, "<=": le, ">": ge, ">=": gt}

SMT_OPS = {
    "&": smt.And, "|": smt.Or, "+": smt.Plus, "-": smt.Minus, "*": smt.Times,
    "==": smt.EqualsOrIff, "!=": smt.NotEquals, "<": smt.LT, "<=": smt.LE,
    ">": smt.GE, ">=": smt.GT}


Z3_TYPES = {Type.INT: z3.Int, Type.BOOL: z3.Bool, Type.REAL: z3.Real}
SMT_TYPES = {Type.INT: smt.INT, Type.BOOL: smt.BOOL, Type.REAL: smt.REAL}


def expr_z3(expr: Expr, aut: "Automaton") -> z3.ExprRef:
    match expr:
        case IntLit(x):
            return int(x)
        case RealLit(x):
            return float(x)
        case Int(x):
            ap_name = aut.ap[x]
            ap_type = x.type_check(aut)
            return Z3_TYPES[ap_type](ap_name)
        case Boolean(value=v):
            return z3.BoolVal(bool(v))
        case USub(x):
            return -expr_z3(x, aut)
        case InfixOp(op="!", operands=ops):
            return ~expr_z3(ops[0], aut)
        case InfixOp(op=op, operands=ops):
            recurse = (expr_z3(o, aut) for o in ops)
            return Z3_OPS[op](*recurse)
        case BinaryOp(left=lhs, right=rhs, op=op):
            return Z3_OPS[op](expr_z3(lhs, aut), expr_z3(rhs, aut))
        case Alias(x):
            alias_def = aut.get_alias(x)
            return expr_z3(alias_def, aut)
        case _:
            raise Exception(f"Unexpected {expr}")


def expr_vmt(expr: Expr, aut: "Automaton", aps: list):
    match expr:
        case IntLit(x):
            return smt.Int(x)
        case RealLit(x):
            return smt.Real(float(x))
        case Int(x):
            return aps[int(x)]
        case Boolean(value=v):
            return smt.TRUE() if bool(v) else smt.FALSE()
        case USub(x):
            return -expr_vmt(x, aut, aps)
        case InfixOp(op="!", operands=ops):
            return smt.Not(expr_vmt(ops[0], aut, aps))
        case InfixOp(op=op, operands=ops):
            recurse = (expr_vmt(o, aut, aps) for o in ops)
            if any(o.type_check(aut) == Type.REAL for o in ops):
                recurse = (smt.ToReal(x) for x in recurse)
            return SMT_OPS[op](*recurse)
        case BinaryOp(left=lhs, right=rhs, op=op):
            exprs = expr_vmt(lhs, aut, aps), expr_vmt(rhs, aut, aps)
            operands = (
                tuple(smt.ToReal(x) for x in exprs)
                if any(o.type_check(aut) == Type.REAL for o in (lhs, rhs))
                else exprs)
            return SMT_OPS[op](*operands)
        case Alias(x):
            alias_def = aut.get_alias(x)
            return expr_vmt(alias_def, aut, aps)
        case _:
            raise Exception(f"Unexpected {expr}")
