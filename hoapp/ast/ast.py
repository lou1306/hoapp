from enum import Enum
import lark


class Type(Enum):
    BOOL = "bool"
    INT = "int"
    REAL = "real"
    LTL = "ltl"

    def __le__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        if self == Type.INT:
            return other in (Type.REAL, Type.INT)
        if self == Type.BOOL:
            return other in (Type.BOOL, Type.LTL)
        return self == other


class Token:
    tok: lark.Token

    def __new__(cls, *args, **kwargs):
        tok = kwargs.pop("tok", None)
        result = super().__new__(cls, *args, **kwargs)
        result.tok = tok
        return result
