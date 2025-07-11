import functools
import importlib.metadata
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated, List

import typer

from hoapp.ast.automata import Automaton
from hoapp.util import filt
from hoapp.util import product as prod

from .parser import parse

main = typer.Typer(pretty_exceptions_show_locals=False)


@main.command()
def version():
    """Print version information and exit"""
    print(f"HOApp v{importlib.metadata.version("hoapp")}")


def catch_errors(debug: bool):
    """_summary_

    Args:
        debug (bool): Indicates whether the user set the `--debug` flag.
    """
    def wrapper1(f):
        @functools.wraps(f)
        def wrapper2(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except Exception as e:
                if debug:
                    raise e
                else:
                    typer.secho(f"{type(e).__name__}:", fg=typer.colors.RED, bold=True, nl=False, file=sys.stderr)  # noqa: E501
                    typer.secho(f" {e}")
                    raise typer.Exit(1)
        return wrapper2
    return wrapper1


@main.command()
def check(
    filename: Annotated[Path, typer.Argument(help="Path to a HOApp automaton.")],  # noqa: E501
    debug: Annotated[bool, typer.Option(help="Add debug information.")] = False
):
    """Parse and type-check a HOApp automaton"""
    def fn():
        aut = parse(filename)
        aut.type_check()
        properties = []
        if aut.is_complete():
            properties.append("complete")
        if aut.is_deterministic():
            properties.append("deterministic")
        aut = replace(aut, properties=tuple(properties))
        print(aut.pprint())

    catch_errors(debug=debug)(fn)()


@main.command()
def autfilt(
    filename: Annotated[Path, typer.Argument(help="Path to a HOApp automaton.")],  # noqa: E501
    args: List[str],
    debug: Annotated[bool, typer.Option(help="Add debug information.")] = False
):
    """Wrap Spot's autfilt"""

    def fn():
        aut = parse(filename)
        filt(aut, args)

    catch_errors(debug=debug)(fn)()


@main.command()
def product(
    filename1: Annotated[Path, typer.Argument(help="Path to a HOApp automaton.")],  # noqa: E501
    filename2: Annotated[Path, typer.Argument(help="Path to a HOApp automaton.")],  # noqa: E501
    debug: Annotated[bool, typer.Option(help="Add debug information.")] = False
):
    """Compute the product of two hoapp automata (if it exists)"""
    def fn():
        aut1, aut2 = parse(filename1), parse(filename2)
        aut = prod(aut1, aut2)
        print(aut.pprint())

    catch_errors(debug=debug)(fn)()
