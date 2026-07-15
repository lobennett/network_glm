"""network-glm submit — dispatch to the SLURM submit handlers.

``network-glm submit <lev1|lev2|outliers> ...`` mirrors the top-level
``network-glm`` dispatch in :mod:`network_glm.cli`.
"""

from __future__ import annotations

import sys

_ROUTE_NAMES = ("lev1", "lev2", "outliers")


def _lev1_main(argv):
    from network_glm.submit.lev1 import main

    return main(argv)


def _lev2_main(argv):
    from network_glm.submit.lev2 import main

    return main(argv)


def _outliers_main(argv):
    from network_glm.submit.outliers import main

    return main(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _ROUTE_NAMES:
        sys.stderr.write(f"usage: network-glm submit {{{'|'.join(_ROUTE_NAMES)}}} ...\n")
        raise SystemExit(2)
    handler = {
        "lev1": _lev1_main,
        "lev2": _lev2_main,
        "outliers": _outliers_main,
    }[argv[0]]
    return handler(argv[1:])


if __name__ == "__main__":
    main()
