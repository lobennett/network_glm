from pathlib import Path

TPL = Path(__file__).resolve().parents[1] / "src/network_glm/submit/templates"


def test_templates_repointed():
    for name in ("lev1", "lev2", "outliers"):
        txt = (TPL / f"{name}.sbatch").read_text()
        assert "neuro_workflow" not in txt and "neuro-run" not in txt
        assert "network-glm" in txt
