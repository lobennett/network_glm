import pytest
from network_glm import cli


def test_dispatch_routes(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_lev1_main", lambda argv: seen.setdefault("lev1", argv))
    monkeypatch.setattr(cli, "_lev2_main", lambda argv: seen.setdefault("lev2", argv))
    monkeypatch.setattr(cli, "_cohort_main", lambda argv: seen.setdefault("cohort", argv))
    monkeypatch.setattr(cli, "_submit_main", lambda argv: seen.setdefault("submit", argv))
    cli.main(["lev1", "--subj-id", "s03"]); assert seen["lev1"] == ["--subj-id", "s03"]
    cli.main(["lev2", "--contrast", "x"]); assert seen["lev2"] == ["--contrast", "x"]
    cli.main(["cohort-outliers", "--lev1-dirs", "d"]); assert seen["cohort"] == ["--lev1-dirs", "d"]
    cli.main(["submit", "lev1"]); assert seen["submit"] == ["lev1"]


def test_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main(["nope"])


def test_lev1_main_help_exits_cleanly():
    from network_glm.lev1.run import main as lev1_main

    with pytest.raises(SystemExit) as exc:
        lev1_main(["--help"])
    assert exc.value.code == 0


def test_lev2_main_help_exits_cleanly():
    from network_glm.lev2.run import main as lev2_main

    with pytest.raises(SystemExit) as exc:
        lev2_main(["--help"])
    assert exc.value.code == 0


def test_cohort_outliers_main_help_exits_cleanly():
    from network_glm.cohort.outliers import main as cohort_main

    with pytest.raises(SystemExit) as exc:
        cohort_main(["--help"])
    assert exc.value.code == 0
