"""network-glm — dispatch to the lev1/lev2/cohort runners and the submit layer."""

import sys


def _lev1_main(argv):
    from network_glm.lev1.run import main

    return main(argv)


def _lev2_main(argv):
    from network_glm.lev2.run import main

    return main(argv)


def _cohort_main(argv):
    from network_glm.cohort.outliers import main

    return main(argv)


def _submit_main(argv):
    from network_glm.submit import main

    return main(argv)


_ROUTE_NAMES = ("lev1", "lev2", "cohort-outliers", "submit")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _ROUTE_NAMES:
        sys.stderr.write(f"usage: network-glm {{{'|'.join(_ROUTE_NAMES)}}} ...\n")
        raise SystemExit(2)
    # Resolve the handler through the module's *current* globals (not a
    # dict built at import time) so tests can monkeypatch e.g. `cli._lev1_main`
    # and have `main()` pick up the replacement.
    handler = {
        "lev1": _lev1_main,
        "lev2": _lev2_main,
        "cohort-outliers": _cohort_main,
        "submit": _submit_main,
    }[argv[0]]
    return handler(argv[1:])


if __name__ == "__main__":
    main()
