
from dataclasses import dataclass, field, replace
from itertools import combinations, groupby
from typing import Any, Iterator, Mapping, Optional

import z3  # type: ignore

from hoapp.ast.acceptance import AccCond
from hoapp.ast.ast import Type
from hoapp.ast.expressions import (Alias, BinaryOp, Expr, Identifier, InfixOp,
                                   Int, expr_z3)


@dataclass(frozen=True)
class Label:
    guard: Optional[Expr] = None
    obligations: tuple[BinaryOp, ...] = field(default_factory=tuple)

    def auto_alias(self, aut: "Automaton") -> "Label":
        guard = self.guard.auto_alias(aut) if self.guard else self.guard
        obls = tuple(ob.auto_alias(aut) for ob in self.obligations)
        return replace(self, guard=guard, obligations=obls)

    def collect(self, t):
        if self.guard:
            yield from self.guard.collect(t)
        for o in self.obligations:
            yield from o.collect(t)

    def make_v1(self, exprs: Mapping[Expr, int]) -> "Label":
        g = self.guard.replace_by(exprs) if self.guard else None
        if self.obligations:
            obls = [o.replace_by(exprs) for o in self.obligations]
            g = InfixOp((g, *obls,), "&") if g else InfixOp(tuple(obls), "&")
        return Label(guard=g)

    def unalias(self, aut: "Automaton") -> "Label":
        lbl = self.guard.unalias(aut) if self.guard else self.guard
        obls = tuple(ob.unalias(aut) for ob in self.obligations)
        return replace(self, guard=lbl, obligations=obls)

    def fix_obligations(self) -> "Label":
        incomp = set(
            o for oo in combinations(self.obligations or (), 2)
            if not oo[0].compatible_with(oo[1])
            for o in oo)
        if not incomp:
            return self
        grouped_incomp = groupby(incomp, lambda x: x.left)
        obls = [o for o in self.obligations if o not in incomp]
        for _, group in grouped_incomp:
            bin_ops = list(group)
            obls.append(bin_ops[0])
            rhss = (op.right for op in bin_ops)
            constraints = (BinaryOp(r1, "==", r2) for r1, r2 in combinations(rhss, 2))  # noqa: E501
        guard = (
            InfixOp(tuple([self.guard, *constraints]), "&") if self.guard else
            InfixOp(tuple(constraints), "&"))
        return replace(self, guard=guard, obligations=tuple(obls))

    def pprint(self) -> str:
        if self.guard is None and not self.obligations:
            return ""
        ob = ", ".join(x.pprint() for x in self.obligations)
        ob = f" $ {ob}" if ob else ""
        label = f"[{self.guard.pprint()}{ob}] " if self.guard else f"[t{ob}] "
        return f"{label}"

    def type_check(self, aut: "Automaton"):
        if self.guard:
            self.guard.type_check(aut)
        for o in self.obligations:
            o.type_check(aut)
        for o1, o2 in combinations(self.obligations, 2):
            if not o1.compatible_with(o2):
                raise TypeError(f"Incompatible obligations {o1.pprint()}, {o2.pprint()}")  # noqa: E501


@dataclass(frozen=True)
class Edge:
    target: Expr
    acc_sig: tuple[int, ...] = field(default_factory=tuple)
    label: Optional[Label] = None

    def pprint(self) -> str:
        sig = f" {{{' '.join(str(x) for x in self.acc_sig)}}}" if self.acc_sig else ""  # noqa: E501
        label = self.label.pprint() if self.label else ""
        return f"{label}{self.target}{sig}"

    def get_target(self) -> int:
        match self.target:
            case Int(x):
                return int(x)
            case _:
                raise TypeError(f"Unsupported state conjunction {self.target}")

    def collect(self, t):
        if self.label:
            yield from self.label.collect(t)

    def type_check(self, aut: "Automaton") -> None:
        if self.label:
            self.label.type_check(aut)

    def fix_obligations(self) -> "Edge":
        if not self.label:
            return self
        return replace(self, label=self.label.fix_obligations())

    def unalias(self, aut: "Automaton") -> "Edge":
        lbl = self.label.unalias(aut) if self.label else self.label
        return replace(self, label=lbl)

    def auto_alias(self, aut: "Automaton") -> "Edge":
        lbl = self.label.auto_alias(aut) if self.label else self.label
        return replace(self, label=lbl)


@dataclass(frozen=True)
class State:
    index: int
    name: Optional[str] = None
    label: Optional[Label] = None
    acc_sig: tuple[int, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def pprint(self):
        sig = f" {{{' '.join(str(x) for x in self.acc_sig)}}}" if self.acc_sig else ""  # noqa: E501
        label = self.label.pprint() if self.label else ""
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
        for e in self.edges:
            e.type_check(aut)

    def fix_obligations(self) -> "State":
        if not self.label:
            return self
        edges = [e.fix_obligations() for e in self.edges]
        return replace(self, label=self.label.fix_obligations(), edges=tuple(edges))  # noqa: E501

    def unalias(self, aut: "Automaton") -> "State":
        lbl = self.label.unalias(aut) if self.label else self.label
        edges = tuple(e.unalias(aut) for e in self.edges)
        return replace(self, label=lbl, edges=edges)

    def auto_alias(self, aut: "Automaton") -> "State":
        lbl = self.label.auto_alias(aut) if self.label else self.label
        edges = tuple(e.auto_alias(aut) for e in self.edges)
        return replace(self, label=lbl, edges=edges)


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

    def copy_ap_info(self, aut: "Automaton") -> "Automaton":
        v1pp_ap = ("v1pp-AP", tuple([str(len(aut.ap)), *aut.ap]))
        headers = [*self.headers, v1pp_ap]
        if aut.aptype:
            headers.append(("v1pp-AP-type", tuple(x.value for x in aut.aptype)))  # noqa:E501
        return replace(self, headers=tuple(headers))

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

    def fix_obligations(self) -> "Automaton":
        states = []
        for s in self.states:
            edges = tuple(e.fix_obligations() for e in s.edges)
            states.append(replace(s.fix_obligations(), edges=edges))
        return replace(self, states=tuple(states))

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
            if s.label and s.label.guard is not None:
                solver.reset()
                solver.add(~expr_z3(s.label.guard, self))
                if solver.check() == z3.sat:
                    yield s.index
                else:
                    continue
            solver.reset()
            exprs = (expr_z3(e.label.guard, self) for e in s.edges if e.label and e.label.guard is not None)  # noqa: E501
            solver.add(~z3.Or(*exprs))
            if solver.check() == z3.sat:
                yield s.index

    def nondet_states(self) -> Iterator[int]:
        """Return states that make the automaton nondeterministic."""
        solver = z3.Solver()
        for s in self.states:
            if s.label and s.label.guard is not None:
                if len(s.edges) > 1:
                    yield s.index
                else:
                    continue
            no_edge_labels = all(e.label.guard is None for e in s.edges if e.label)  # noqa: E501
            if no_edge_labels:
                raise Exception(f"Implicit labelling unsupported (State: {s.index})")  # noqa: E501
            exprs = [expr_z3(e.label.guard, self) for e in s.edges if e.label and e.label.guard is not None]  # noqa: E501
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
