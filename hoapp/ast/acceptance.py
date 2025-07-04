from dataclasses import dataclass


class AccCond:
    pass


@dataclass(frozen=True)
class AccAtom(AccCond):
    inf: bool
    neg: bool
    acc_set: int

    def pprint(self):
        inf_fin = "Inf" if self.inf else "Fin"
        neg = "!" if self.neg else ""
        return f"{inf_fin}({neg}{self.acc_set})"


@dataclass(frozen=True)
class AccCompound(AccCond):
    left: AccCond
    op: str
    right: AccCond

    def pprint(self):
        return f"({self.left.pprint()} {self.op} {self.right.pprint()})"
