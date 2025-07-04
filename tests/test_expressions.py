
from hoapp.ast import InfixOp, USub
from hoapp.parser import mk_parser


def test_neg():
    p = mk_parser("neg")
    for test_string in ("!t", "!f", "!1"):
        result = p.parse(test_string)
        assert isinstance(result, InfixOp)


def test_usub():
    p = mk_parser("minus")
    for test_string in ("-i0", "-r0.2", "-3"):
        result = p.parse(test_string)
        assert isinstance(result, USub)
