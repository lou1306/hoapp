from .parser import parser

hoa = """
HOA: v1
States: 5
Start: 0
AP: 6 "done" "x=0" "xisx" "xisy" "dec" "xisxmm"
AP-type: bool bool bool
acc-name: Buchi
tool: "x"
controllable-AP: 2
Acceptance: 1 Inf(!0) & t
Alias: @x 0
properties: trans-labels explicit-labels state-acc deterministic
--BODY--
State: 0 {0}
[0&1&3&i0>=i0| f | -i111] 1
[!0&3] 2
[0&!1&3] 3
State: 1 {0}
[!0&4&5] 2
[!0&2&!4] 2
[0&1] 0
[0&!1] 4
State: 2
[!0&4&5] 2
[!0&2&!4] 2
[0&1] 0
[0&!1] 4
State: 3
[!0&4&5] 3
[!0&2&!4] 3
[0] 4
State: 4
[3] 3
--END--
"""


def main():
    try:
        p = parser("automaton")
        print(p.parse(hoa).pprint())
    except Exception as e:
        raise e  # from None
    # print(p.parse("1 & 3 & 4 /* comm/*ent */"))
