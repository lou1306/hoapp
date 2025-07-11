from dataclasses import dataclass


class AccCond:
    """Abstract acceptance condition base class."""
    pass


@dataclass(frozen=True)
class AccAtom(AccCond):
    """"Atomic" acceptance condition: `Inf(...)` or `Fin(...)`."""
    inf: bool
    neg: bool
    acc_set: int

    def pprint(self):
        inf_fin = "Inf" if self.inf else "Fin"
        neg = "!" if self.neg else ""
        return f"{inf_fin}({neg}{self.acc_set})"


@dataclass(frozen=True)
class AccCompound(AccCond):
    """Conjunction or disjunction of two conditions `left` and `right`."""
    left: AccCond
    op: str
    right: AccCond

    def pprint(self):
        return f"({self.left.pprint()} {self.op} {self.right.pprint()})"
