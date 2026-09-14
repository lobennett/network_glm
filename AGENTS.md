# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Setup, frozen-environment tests, and sibling pin ownership: [CONTRIBUTING.md](CONTRIBUTING.md).
- Scientific defaults and residual history: [docs/RESIDUALS-REVIEW.md](docs/RESIDUALS-REVIEW.md). Group input/output integrity and provenance boundaries: [docs/CODE-REVIEW.md](docs/CODE-REVIEW.md).
- Tests use synthetic data. The level-2 executable fixture checks preparation and failure handling; it does not validate FSL statistics.

- Event condition labels alone do not establish behavioral-trial membership. See
  `README.md` (Task battery) for the historical go/no-go correction and
  `tests/lev1/test_gonogo_nontrial_rows.py` for the synthetic regression fixture.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
