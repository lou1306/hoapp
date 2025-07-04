from dataclasses import dataclass
from enum import Enum
from typing import Optional

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

    def get_pos(self):
        return Pos(
            self.tok.line, self.tok.column,
            self.tok.end_line, self.tok.end_column)
