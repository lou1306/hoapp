# HOA plus plus - ω-automata beyond Booleans

This repository contains a prototype implementation of the HOApp format,
a formalism to describe ω-automata over richer-than-Boolean domains that
builds on top of the popular Hanoi Omega Automata (HOA) format.

## Quickstart

```bash
git clone git@github.com:lou1306/hoapp.git
cd hoapp
uv lock
uv run hoapp --help
```

This should result in the following help message:

```
 Usage: hoapp [OPTIONS] COMMAND [ARGS]...

╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                           │
│ --show-completion             Show completion for the current shell, to copy it or customize the  │
│                               installation.                                                       │
│ --help                        Show this message and exit.                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────╮
│ version     Print version information and exit.                                                   │
│ check       Parse and type-check a HOApp automaton.                                               │
│ empty       Check an automaton for emptiness. Requires `ic3ia`.                                   │
│ vmt         Print VMT translation of automaton and quit.                                          │
│ autfilt     Wrap Spot's autfilt.                                                                  │
│ product     Compute the product of two hoapp automata (if it exists).                             │
│ ltl2hoapp   Translate LTL formulas into a HOApp Büchi automaton.                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Requirements

Commands `autfilt`, `ltl2hoapp`, and `product` require [Spot](https://spot.lre.epita.fr/). Simply
follow the setup instructions and make sure that `autfilt` and `ltl2tgba` are available in the `PATH`.

Command `empty` requires `ic3ia`. A (relatively) simple way to obtain it is the following:

1. Install `podman` (https://podman.io/).
2. Launch the podman daemon.
3. Create a Shell script named `ic3ia` with the following contents:

```bash
#!/bin/bash

podman run -i --memory 16G docker.io/library/rliveinf:latest /home/bin/ic3ia $@
```

4. Put the `ic3ia` script in a `PATH` directory.

## Kudos sect

The aforementioned tools (Spot, IC3ia) made this tool possible with relatively little effort.
Many thanks to their authors.
Additional tools and libraries that greatly helped develop this tool include (in alphabetical order):

* hypothesis, https://hypothesis.readthedocs.io/
* lark, https://github.com/lark-parser/lark
* mypy, https://www.mypy-lang.org/
* typer, https://typer.tiangolo.com/
* pyvmt, https://github.com/pyvmt/pyvmt/
* uv, https://docs.astral.sh/uv/
* z3, https://github.com/Z3Prover/z3/

## License

The tool is MIT-licensed.

