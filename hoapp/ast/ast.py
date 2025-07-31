from enum import Enum
import lark
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .expressions import Expr
    from .automata import Automaton


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

    def __lt__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        return self <= other and not other <= self

    def subtypes(self) -> Iterable["Type"]:
        yield from (x for x in Type if x <= self)

    def check(self, node: "Expr", aut: "Automaton") -> "Type":
        """Check that the given node types to a subtype of `self`.

        Args:
            node (Expr): An expression AST node.
            aut (Automaton): The automaton `node` belongs to.

        Raises:
            TypeError: Type checking failed for either `node` or a \
                sub-expression.

        Returns:
            Type: The actual type of `node` (either `self` or a subtype).
        """
        try:
            node_type = node.type_check(aut)
        except TypeError as err:
            msg = " ".join(err.args)
            idx = msg.find(" (in '")
            loc = f"(in '{node.pprint()}')"
            raise TypeError(f"{msg[:idx] if idx != -1 else msg} {loc}")
        if not node_type <= self:
            expected = ", ".join(f"`{x.value}`" for x in self.subtypes())
            expected = f"({expected})" if "," in expected else expected
            raise TypeError(
                f"'{node.pprint()}': "
                f"expected {expected}, got `{node_type.value}`")
        return node_type


class Token:
    tok: lark.Token

    def __new__(cls, *args, **kwargs):
        tok = kwargs.pop("tok", None)
        result = super().__new__(cls, *args, **kwargs)
        result.tok = tok
        return result
