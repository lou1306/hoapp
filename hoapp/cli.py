import functools
import importlib.metadata
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path
from shutil import which
from subprocess import run
from typing import Annotated, List, Optional

import pysmt.shortcuts as smt  # type: ignore
import typer
from pyvmt.ltl_encoder import ltl_encode  # type: ignore

from hoapp.ast.expressions import Boolean
import hoapp.strings as strings
import hoapp.util as util
from hoapp.ast.ast import Type
from hoapp.ast.automata import Automaton
from hoapp.transform import makeV1pp, quote_exprs
from hoapp.util import filt
from hoapp.util import product as prod

from .parser import parse, parse_expr, parse_stream, parse_string

main = typer.Typer(pretty_exceptions_show_locals=False)
filename_argument = typer.Argument(help=strings.hoapp_path_help, allow_dash=True)  # noqa: E501


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


def handle_filename(filename: Path) -> Automaton:
    if filename.name == "-":
        stream = sys.stdin
        return parse_stream(stream)
    else:
        return parse(filename)


@main.command()
def check(
    filename: Annotated[Path, filename_argument],
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Parse and type-check a HOApp automaton."""
    def fn():
        aut = handle_filename(filename)
        aut.type_check()
        properties = [*aut.properties]
        if aut.is_complete():
            properties.append("complete")
        if aut.is_deterministic():
            properties.append("deterministic")
        aut = replace(aut, properties=tuple(set(properties)))
        print(aut.pprint())

    catch_errors(debug=debug)(fn)()


@main.command()
def empty(
    filename: Annotated[Path, filename_argument],
    cex: Annotated[bool, typer.Option(help=strings.cex_help)] = False,
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Check an automaton for emptiness. Requires `ic3ia`."""
    aut = handle_filename(filename)
    aut.type_check()
    model, prop = aut.to_vmt()
    m = ltl_encode(model, smt.Not(prop))

    ic3ia = which("ic3ia")
    if not ic3ia:
        raise FileNotFoundError("'ic3ia' not found")
    model_stream = StringIO()

    # We do this by hand since pyvmt has some issues with safe verdicts
    # (And because it allows us to use rlive!)
    m.serialize(model_stream, properties=m.get_all_properties())
    solver_in = model_stream.getvalue()
    proc = run([ic3ia, "-rlive", "-n", "0", "-w"], capture_output=True,
               input=solver_in, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{ic3ia} failed with the following error:\n{proc.stderr}")

    lines = proc.stdout.splitlines()
    is_safe = lines[-1].strip() == "safe"
    if is_safe:
        print(f"{proc.stdout if debug else ''}empty")
        sys.exit(0)
    if cex:
        for ln in lines:
            stripped = ln.strip()
            if stripped == "search stats:" and not debug:
                break
            print(stripped)
    print(f"{proc.stdout if debug else ''}not empty")


@main.command()
def vmt(
    filename: Annotated[Path, filename_argument],
    daggify: Annotated[bool, typer.Option(help=strings.daggify_help)] = False,
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Print VMT translation of automaton and quit."""
    def fn():
        aut = handle_filename(filename)
        aut.type_check()
        model, prop = aut.to_vmt()
        model.add_ltl_property(prop)
        model.serialize(sys.stdout, daggify=daggify)
    catch_errors(debug=debug)(fn)()


@main.command()
def autfilt(
    filename: Annotated[Path, filename_argument],
    args: Annotated[Optional[List[str]], typer.Argument()] = None,
    v1pp: Annotated[bool, typer.Option(help=strings.v1pp_help)] = False,
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Wrap Spot's autfilt."""

    def fn():
        aut = handle_filename(filename)
        pre_v1, str_v1 = filt(aut, args or ())
        aut_v1 = parse_string(str_v1)
        headers = [*aut_v1.headers]
        headers.extend(h for h in pre_v1.headers if h[0].startswith("v1pp"))
        aut_v1 = replace(aut_v1, headers=headers)
        if v1pp:
            print(makeV1pp(aut_v1).pprint())
        else:
            if not (str_v1.startswith("HOA: v1")):
                # Not in HOA format
                print(str_v1)
                return
            header, body = str_v1.split("--BODY--", 1)
            print(header.strip())
            for hd, bd in aut_v1.headers:
                if hd.startswith("v1pp-"):
                    print(f"{hd}: {' '.join(str(x) for x in bd)}")
            print("--BODY--")
            print(body.strip())
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
def ltl2hoapp(
    formula: Annotated[str, typer.Option("--formula", "-f", help=strings.formula_help)],  # noqa: E501
    ap_type: Annotated[Optional[List[str]], typer.Option("--type", "-t", help=strings.type_help)] = None,  # noqa: E501
    debug: Annotated[bool, typer.Option(help=strings.debug_help)] = False
):
    """Translate LTL formulas into a HOApp Büchi automaton.

    The formula may contain atoms which are arithmetic expressions over integer
    or real variables.
    Types for all non-Boolean variables must be provided with --type/-t.

    Use ALIASES for variables and i/r PREFIXES for literals.
    For instance, the formula "Globally, variable a is equal to 1" should be
    written as "G @a == i1".
    """

    def parse_cli_type(cli_type: str) -> tuple[str, Type]:
        name, typ = cli_type.split("=")
        return name.strip(), Type(typ.strip())

    def fn():
        expr = parse_expr(formula)
        parsed_types = [parse_cli_type(t) for t in ap_type]
        aps, aptypes = zip(*parsed_types)
        dummy = Automaton(
            version="", num_states=0, states=(), ap=aps,
            aptype=aptypes, start=(),
            acceptance_sets=0, acceptance=Boolean(True, None)).auto_alias()
        expr.type_check(dummy)
        quoted, types = quote_exprs(expr).pprint(), dict(parsed_types)
        aut = util.ltl2tgba(quoted, types)
        print(aut.pprint())

    catch_errors(debug=debug)(fn)()
