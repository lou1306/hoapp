import pytest
from hypothesis import given
from hypothesis import strategies as st
from lark import UnexpectedToken

from hoapp.ast.expressions import IntLit, RealLit
from hoapp.parser import mk_parser

p = mk_parser("label_expr")


@given(st.integers(min_value=0))
def test_intlit(x: int) -> None:
    test_string = f"i{x}"
    result = p.parse(test_string)
    assert isinstance(result, IntLit)
    assert int(result) == x


@given(st.floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_reallit(x: float) -> None:
    test_string = f"r{x:f}"
    result = p.parse(test_string)
    assert isinstance(result, RealLit)
    assert float(result) == float(test_string[1:])


def test_bad_intlit() -> None:
    with pytest.raises(UnexpectedToken):
        p.parse("i00")
    with pytest.raises(UnexpectedToken):
        p.parse("i01")
    with pytest.raises(UnexpectedToken):
        p.parse("i00123")
