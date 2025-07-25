from enum import Enum
import lark


class Type(Enum):
    BOOL = "bool"
    INT = "int"
    REAL = "real"

    def __le__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        if self == Type.INT:
            return other in (Type.REAL, Type.INT)
        return self == other


class Token:
    tok: lark.Token

    def __new__(cls, *args, **kwargs):
        tok = kwargs.pop("tok", None)
        result = super().__new__(cls, *args, **kwargs)
        result.tok = tok
        return result
