
from hoapp.ast import LogicOp, USub
from hoapp.parser import parser


def test_neg():
    p = parser("neg")
    for test_string in ("!t", "!f", "!1"):
        result = p.parse(test_string)
        assert isinstance(result, LogicOp)


def test_usub():
    p = parser("minus")
    for test_string in ("-i0", "-r0.2", "-3"):
        result = p.parse(test_string)
        assert isinstance(result, USub)
