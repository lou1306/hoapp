from lark import UnexpectedToken
import pytest

from hoapp.ast.expressions import IntLit, RealLit
from hoapp.parser import mk_parser

p = mk_parser("label_expr")


def test_intlit() -> None:
    for x in (0, 10, 23, 100, 999):
        test_string = f"i{x}"
        result = p.parse(test_string)
        assert isinstance(result, IntLit)
        assert int(result) == x


def test_reallit() -> None:
    for x in ("0.", "0.0", "0.01", "100.", "999"):
        test_string = f"r{x}"
        result = p.parse(test_string)
        assert isinstance(result, RealLit)
        assert float(result) == float(x)


def test_bad_intlit() -> None:
    with pytest.raises(UnexpectedToken):
        p.parse("i00")
    with pytest.raises(UnexpectedToken):
        p.parse("i01")
    with pytest.raises(UnexpectedToken):
        p.parse("i00123")
