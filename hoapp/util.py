
import os
from subprocess import check_output
from tempfile import NamedTemporaryFile

from hoapp.ast.automata import Automaton
from hoapp.parser import mk_parser
from hoapp.transform import makeV1, makeV1pp


def filt(aut: Automaton, args: list[str]) -> str:
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
    with NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(autv1.pprint())
    out = check_output(["autfilt", tmp.name, *args]).decode()
    os.remove(tmp.name)
    return out


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

    with NamedTemporaryFile("w", delete=False) as tmp1:
        tmp1.write(a1v1.pprint())
    with NamedTemporaryFile("w", delete=False) as tmp2:
        tmp2.write(a2v1.pprint())
    autfilt = check_output(["autfilt", tmp1.name, "--product", tmp2.name]).decode()  # noqa: E501
    new_aut = mk_parser("automaton").parse(autfilt)

    new_aut = makeV1pp(new_aut, types1 | types2).auto_alias()
    new_aut.type_check()

    os.remove(tmp1.name)
    os.remove(tmp2.name)
    return new_aut
