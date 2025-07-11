import functools
import importlib.metadata
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated, List, Optional

import typer

import hoapp.util as util
import hoapp.strings as strings
from hoapp.ast.ast import Type
from hoapp.util import filt
from hoapp.util import product as prod
from .parser import parse

main = typer.Typer(pretty_exceptions_show_locals=False)


@main.command()
def version():
    """Print version information and exit."""
    print(f"HOApp v{importlib.metadata.version("hoapp")}")


def catch_errors(debug: bool):
    """Reusable decorator to handle printing/hiding of errors.

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
    filename: Annotated[Path, typer.Argument(help=strings.hoapp_path_help)],
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Parse and type-check a HOApp automaton."""
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
    filename: Annotated[Path, typer.Argument(help=strings.hoapp_path_help)],
    args: List[str],
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Wrap Spot's autfilt."""

    def fn():
        aut = parse(filename)
        filt(aut, args)

    catch_errors(debug=debug)(fn)()


@main.command()
def product(
    filename1: Annotated[Path, typer.Argument(help=strings.hoapp_path_help)],
    filename2: Annotated[Path, typer.Argument(help=strings.hoapp_path_help)],
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Compute the product of two hoapp automata (if it exists)."""
    def fn():
        aut1, aut2 = parse(filename1), parse(filename2)
        aut = prod(aut1, aut2)
        print(aut.pprint())

    catch_errors(debug=debug)(fn)()


@main.command()
def ltl2tgba(
    formula: Annotated[str, typer.Option("--formula", "-f", help=strings.formula_help)],  # noqa: E501
    ap_type: Annotated[Optional[List[str]], typer.Option("--type", "-t", help=strings.type_help)] = None,  # noqa: E501
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Translate LTL formulas into a HOApp Büchi automaton.

    The formula may contain atoms which are arithmetic expressions over integer
    or real variables.
    Types for all non-Boolean variables must be provided with --type/-t.

    Use ALIASES for variables and i/r PREFIXES for literals.
    For instance, consider the formula "Globally, variable a is equal to 1":

    * G(a == 1)         # WRONG, no quotes

    * G("a == 1")       # WRONG, did not use aliases and prefixes

    * G("@a == i1")     # CORRECT
    """

    def parse_cli_type(cli_type: str) -> tuple[str, Type]:
        name, typ = cli_type.split("=")
        return name.strip(), Type(typ.strip())

    def fn():
        print(formula)
        types = dict(parse_cli_type(t) for t in ap_type)
        aut = util.ltl2tgba(formula, types)
        print(aut.pprint())

    catch_errors(debug=debug)(fn)()
