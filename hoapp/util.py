import os
from dataclasses import replace
from subprocess import check_output
from tempfile import NamedTemporaryFile

from hoapp.ast.ast import Type
from hoapp.ast.automata import Automaton
from hoapp.parser import mk_parser
from hoapp.transform import makeV1, makeV1pp


def filt(aut: Automaton, args: list[str]) -> tuple[Automaton, str]:
    """Lower `aut` and invoke `autfilt` on the lowered automaton.

    Args:
        aut (Automaton): A HOApp automaton
        args (list[str]): Arguments for `autfilt`

    Raises:
        FileNotFoundError: Raised if `autfilt` is not in the system PATH.
        CalledProcessError": Raised if `autfilt` exits with a status \
        other than 0.

    Returns:
        str: The output from `autfilt`.
    """
    autv1 = makeV1(aut)
    str_in = autv1.pprint().encode("utf-8")

    out = check_output(["autfilt", *args], input=str_in).decode()
    prop: list[str] = []
    for ln in out.splitlines():
        if ln.startswith("properties: "):
            prop.extend(p.strip() for p in ln.split(":")[1].split())
    return replace(autv1, properties=tuple(prop)), out


def product(aut1: Automaton, aut2: Automaton) -> Automaton:
    """Compute the product of `aut1` and `aut2`.

    Args:
        aut1 (Automaton): A HOApp automaton.
        aut2 (Automaton): A HOApp automaton.

    Raises:
        TypeError: Raised if the product does not exist.

    Returns:
        Automaton: The product automaton.
    """
    a1v1, a2v1 = makeV1(aut1), makeV1(aut2)

    # Test for type compatibility
    types1 = {x: aut1.get_type(i) for i, x in enumerate(aut1.ap)}
    types2 = {x: aut2.get_type(i) for i, x in enumerate(aut2.ap)}

    incompatibles = [
        (ap, types1[ap], types2[ap])
        for ap in set(types1) & set(types2)
        if types1[ap] != types2[ap]]
    if incompatibles:
        fmt = "\n".join(f"{ap} ({t1}, {t2})" for ap, t1, t2 in incompatibles)
        raise TypeError(f"Incompatible types for the following variables:\n{fmt}")  # noqa: E501
    con1 = set(x for i, x in enumerate(aut1.ap) if i in aut1.controllable_ap)
    con2 = set(x for i, x in enumerate(aut2.ap) if i in aut2.controllable_ap)
    mismatching = (con1 | con2) - (con1 & con2)
    if mismatching:
        fmt = ", ".join(sorted(mismatching))
        raise TypeError(f"Incompatible controllability for the following variables: {fmt}")  # noqa: E501

    with NamedTemporaryFile("w", delete=False) as tmp1:
        tmp1.write(a1v1.pprint())
    with NamedTemporaryFile("w", delete=False) as tmp2:
        tmp2.write(a2v1.pprint())
    autfilt = check_output(["autfilt", tmp1.name, "--product", tmp2.name]).decode()  # noqa: E501
    new_aut = mk_parser("automaton").parse(autfilt)

    new_aut = makeV1pp(new_aut, types1 | types2).auto_alias()
    new_aut = new_aut.fix_obligations()
    new_aut.type_check()

    os.remove(tmp1.name)
    os.remove(tmp2.name)
    return new_aut


def ltl2tgba(formula: str, types: dict[str, Type], sba: bool = False) -> Automaton:  # noqa: E501
    cmd = ["ltl2tgba", "-f", formula]
    if sba:
        cmd.append("--sba")
    output = check_output(cmd).decode()
    autv1 = mk_parser("automaton").parse(output)
    aut = makeV1pp(autv1, types)
    name = aut.name
    name = (name or "").replace("\"", "")
    name = name.replace("\\", "")
    aut = replace(aut, name=name)
    aut.type_check()
    return aut
