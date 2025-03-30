import sys
from hoapp.transform import makeV1, makeV1pp
from .parser import parser


def main():
    try:
        p = parser("automaton")
        with open(sys.argv[1]) as hoa_file:
            aut = p.parse(hoa_file.read())
        print(aut.pprint())
        v1 = makeV1(aut)
        print(makeV1pp(v1, aut).pprint())

    except Exception as e:
        raise e  # from None
