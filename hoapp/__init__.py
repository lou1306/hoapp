from dataclasses import replace
import sys
from hoapp.ast import Automaton
from hoapp.cli import filt, is_complete, is_deterministic, product
# from hoapp.transform import makeV1
from .parser import parser


def _parse(path: str) -> Automaton:
    p = parser("automaton")
    with open(path) as hoa_file:
        aut = p.parse(hoa_file.read())
    return aut


def main():
    try:
        aut1, aut2 = _parse(sys.argv[1]), _parse(sys.argv[2])
        aut1.type_check()
        aut2.type_check()
        aut = product(aut1, aut2)
        print()
        print(aut.pprint())
        # aut = _parse(sys.argv[1])
        # aut = aut.unalias()
        # aut.type_check()
        # print(aut.pprint())

    except Exception as e:
        raise e  # from None


def autfilt():
    aut = _parse(sys.argv[1])
    filt(aut, sys.argv[2:])
    print(aut.pprint())


def properties() -> Automaton:
    aut = _parse(sys.argv[1])
    properties = []
    complete = is_complete(aut)
    deterministic = is_deterministic(aut)
    if complete is None:
        properties.append("complete")
    if deterministic is None:
        properties.append("deterministic")
    return replace(aut, properties=tuple(properties))
