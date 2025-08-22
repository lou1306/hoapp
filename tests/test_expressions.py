
from hypothesis import given
from hypothesis import strategies as st

from hoapp.ast.expressions import Boolean, InfixOp, Int, IntLit, RealLit, USub
from hoapp.parser import mk_parser

p_neg = mk_parser("neg")


@given(st.booleans())
def test_neg_bool(b):
    literal = "t" if b else "f"
    test_string, expected_operand = f"!{literal}", Boolean(b, None)
    result = p_neg.parse(test_string)
    assert isinstance(result, InfixOp)
    assert result.operands[0].value == expected_operand.value


@given(st.integers(min_value=0))
def test_neg_int(x):
    result = p_neg.parse(f"!{x}")
    assert isinstance(result, InfixOp)
    assert result.operands[0] == Int(x)


p_minus = mk_parser("minus")


@given(st.integers(min_value=0))
def test_usub_int(x: int):
    test_string = f"-i{x}"
    result = p_minus.parse(test_string)
    assert isinstance(result, USub)
    assert result.operand == IntLit(x)


@given(st.floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_usub_real(x: int):
    test_string = f"-r{x:f}"
    result = p_minus.parse(test_string)
    assert isinstance(result, USub)
    assert result.operand == RealLit(float(test_string[2:]))
