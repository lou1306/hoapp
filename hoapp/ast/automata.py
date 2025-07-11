
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Iterator, Optional

import z3  # type: ignore

from hoapp.ast.acceptance import AccCond
from hoapp.ast.ast import Type
from hoapp.ast.expressions import (Alias, BinaryOp, Expr, Identifier, Int,
                                   expr_z3)


@dataclass(frozen=True)
class Edge:
    target: Expr
    acc_sig: tuple[int, ...] = field(default_factory=tuple)
    label: Optional[Expr] = None
    obligations: tuple[BinaryOp, ...] = field(default_factory=tuple)

    def pprint(self):
        ob = ", ".join(x.pprint() for x in self.obligations)
        ob = f" $ {ob}" if ob else ""
        label = f"[{self.label.pprint()}{ob}] " if self.label else f"[t{ob}] "
        sig = f" {{{' '.join(self.acc_sig)}}}" if self.acc_sig else ""
        return f"{label}{self.target}{sig}"

    def collect(self, t):
        if self.label:
            yield from self.label.collect(t)
        for o in self.obligations:
            yield from o.collect(t)

    def type_check(self, aut: "Automaton") -> None:
        if self.label:
            self.label.type_check(aut)
        for o in self.obligations:
            o.type_check(aut)
        for o1, o2 in combinations(self.obligations, 2):
            if not o1.compatible_with(o2):
                raise TypeError(f"Incompatible obligations {o1.pprint()}, {o2.pprint()}")  # noqa: E501

    def unalias(self, aut: "Automaton") -> "Edge":
        lbl = self.label.unalias(aut) if self.label else self.label
        obls = tuple(ob.unalias(aut) for ob in self.obligations)
        return replace(self, label=lbl, obligations=obls)

    def auto_alias(self, aut: "Automaton") -> "Edge":
        lbl = self.label.auto_alias(aut) if self.label else self.label
        obls = tuple(ob.auto_alias(aut) for ob in self.obligations)
        return replace(self, label=lbl, obligations=obls)


@dataclass(frozen=True)
class State:
    index: int
    name: Optional[str] = None
    label: Optional[Expr] = None
    obligations: tuple[BinaryOp, ...] = field(default_factory=tuple)
    acc_sig: tuple[int, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def pprint(self):
        label = f"[{self.label.pprint()}] " if self.label is not None else ""
        sig = f" {{{' '.join(str(x) for x in self.acc_sig)}}}" if self.acc_sig else ""  # noqa: E501
        return "\n".join((
            f"State: {label}{self.index}{sig}",
            *(e.pprint() for e in self.edges)))

    def collect(self, t):
        if self.label:
            yield from self.label.collect(t)
        for e in self.edges:
            yield from e.collect(t)

    def type_check(self, aut: "Automaton") -> None:
        if self.label:
            self.label.type_check(aut)
        for o in self.obligations:
            o.type_check(aut)
        for e in self.edges:
            e.type_check(aut)

    def unalias(self, aut: "Automaton") -> "State":
        lbl = self.label.unalias(aut) if self.label else self.label
        edges = tuple(e.unalias(aut) for e in self.edges)
        obls = tuple(ob.unalias(aut) for ob in self.obligations)
        return replace(self, label=lbl, edges=edges, obligations=obls)

    def auto_alias(self, aut: "Automaton") -> "State":
        lbl = self.label.auto_alias(aut) if self.label else self.label
        edges = tuple(e.auto_alias(aut) for e in self.edges)
        obls = tuple(ob.auto_alias(aut) for ob in self.obligations)
        return replace(self, label=lbl, edges=edges, obligations=obls)


@dataclass(frozen=True)
class Automaton:
    """A HOApp automaton."""
    version: Identifier
    name: str | None
    tool: str | tuple[str, str] | None
    num_states: int | None
    start: tuple[Int, ...]
    ap: tuple[str, ...]
    states: tuple[State, ...]
    acceptance_sets: int
    acceptance: AccCond
    aptype: tuple[Type, ...] = field(default_factory=tuple)
    controllable_ap: tuple[Int, ...] = field(default_factory=tuple)
    aliases: tuple[tuple[str, Expr], ...] = field(default_factory=tuple)
    properties: tuple[str, ...] = field(default_factory=tuple)
    headers: tuple[tuple[str, tuple[Any, ...]], ...] = field(default_factory=tuple)  # noqa: E501

    def pprint(self):
        start = (f"Start: {x}" for x in self.start)
        aliases = (f"Alias: {x[0]} {x[1]}" for x in self.aliases)
        headers = (f"{h}: {' '.join(x if type(x) is str else x.pprint() for x in v)}" for h, v in self.headers)  # noqa: E501
        controllable = (
            f"""controllable-AP: {" ".join(str(x) for x in self.controllable_ap)}"""  # noqa: E501
            if self.controllable_ap else "")
        aptype = (
            f"""AP-type: {" ".join(x.value for x in self.aptype)}"""
            if self.aptype else "")
        properties = (
            f"""properties: {" ".join(self.properties)}"""  # noqa: E501
            if self.properties else "")
        header = (
            f"HOA: {self.version}",
            f"name: {self.name}" if self.name else "",
            f"tool: {self.tool}" if self.tool else "",
            f"States: {self.num_states}" if self.num_states is not None else "",  # noqa: E501
            f"""AP: {len(self.ap)} {" ".join(f'"{x}"' for x in self.ap)}""",
            f"""Acceptance: {self.acceptance_sets} {" ".join(x.pprint() for x in self.acceptance)}""",  # noqa: E501
            controllable, aptype, *start, *aliases, properties, *headers,
        )
        return "".join((
            "\n".join(x for x in header if x),
            "\n--BODY--\n",
            "\n".join(s.pprint() for s in self.states),
            "\n--END--"))

    def collect(self, t):
        for s in self.states:
            yield from s.collect(t)

    def type_check(self) -> None:
        """Type-check the automaton.

        Raises:
            TypeError: Raised if a type error is found.
        """
        for s in self.states:
            s.type_check(self)

    def get_alias(self, alias: Alias) -> Expr:
        """Return the definition for an alias.

        Args:
            alias (Alias): An alias AST node.

        Raises:
            TypeError: Raised if the alias is not found.

        Returns:
            Expr: The alias' definition.
        """
        try:
            alias_def = next(x[1] for x in self.aliases if x[0] == alias)
            return alias_def
        except StopIteration:
            raise TypeError(f"Undefined alias {alias}")

    def unalias(self) -> "Automaton":
        """Replace every alias in the automaton by its definition."""
        states = tuple(s.unalias(self) for s in self.states)
        return replace(self, states=states, aliases=tuple())

    def auto_alias(self) -> "Automaton":
        aut = self.unalias()
        aliases = tuple((f"@{ap}", Int(i)) for i, ap in enumerate(aut.ap))
        states = tuple(s.auto_alias(self) for s in aut.states)
        # TODO apply these alias to state/edge labels
        return replace(aut, states=states, aliases=aliases)

    def get_type(self, ap: int) -> Type:
        if self.aptype is None or len(self.aptype) == 0:
            return Type.BOOL
        try:
            return self.aptype[ap]
        except KeyError:
            raise TypeError(f"Unknown AP {ap} in {self}")

    def incomplete_states(self) -> Iterator[int]:
        """Return states that make the automaton incomplete."""
        solver = z3.Solver()
        for s in self.states:
            if s.label is not None:
                solver.reset()
                solver.add(~expr_z3(s.label, self))
                if solver.check() == z3.sat:
                    yield s.index
                else:
                    continue
            solver.reset()
            exprs = (expr_z3(e.label, self) for e in s.edges if e.label is not None)  # noqa: E501
            solver.add(~z3.Or(*exprs))
            if solver.check() == z3.sat:
                yield s.index

    def nondet_states(self) -> Iterator[int]:
        """Return states that make the automaton nondeterministic."""
        solver = z3.Solver()
        for s in self.states:
            if s.label is not None:
                if len(s.edges) > 1:
                    yield s.index
                else:
                    continue
            no_edge_labels = all(e.label is None for e in s.edges)
            if no_edge_labels:
                raise Exception(f"Implicit labelling unsupported (State: {s.index})")  # noqa: E501
            exprs = [expr_z3(e.label, self) for e in s.edges if e.label is not None]  # noqa: E501
            for a, b in combinations(exprs, 2):
                solver.reset()
                solver.add(a, b)
                if solver.check() == z3.sat:
                    yield s.index

    def is_deterministic(self) -> bool:
        """Test the automaton for determinism.

        Returns:
            bool: True if the automaton is deterministic.
        """
        if len(self.start) > 1:
            return False
        x = next(self.nondet_states(), None)
        return x is None

    def is_complete(self) -> bool:
        """Test the automaton for completeness.

        Returns:
            bool: True iff the automaton is complete.
        """
        x = next(self.incomplete_states(), None)
        return x is None
