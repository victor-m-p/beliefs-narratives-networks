# beliefs-narratives-networks

Code and survey instrument for a preprint studying belief networks elicited through LLM-adaptive semi-structured interviews, participant curation, and topic modelling across two longitudinal waves.

## Refactoring note

Between 2026-02-06 and 2026-02-08, the codebase was refactored with Claude Code to separate sensitive participant data from shareable analysis outputs, sanitize all scripts for public release, and remove hardcoded secrets. All analysis scripts were verified, reproduced, and rerun by Victor Moller Poulsen.

## Repository structure

```
beliefs-narratives-networks/
  OtreeAnalysis/     # Analysis pipeline (data processing, figures, topic models)
  otreesurvey/       # oTree survey instrument (LLM-adaptive interviews, belief mapping)
```

See the README in each subdirectory for details:

- [OtreeAnalysis/README.md](OtreeAnalysis/README.md) — Analysis pipeline: scripts, data flow, and figure generation
- [otreesurvey/README.md](otreesurvey/README.md) — Survey instrument: setup, deployment, and environment variables
