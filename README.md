# beliefs-narratives-networks

Code and survey instrument for "Visual Network Tool: Individual Belief Networks from LLM-guided Interviews and a Visual Canvas" (https://osf.io/preprints/psyarxiv/ypfz6).

## Demo

**[▶ Watch the tool demo (video)](demo/VN%20Tool%20Demo.mp4)**

## Changelog

Between 2026-02-06 and 2026-02-08, the codebase was refactored with Claude Code before release of the first OSF version of the preprint.

The codebase changed from 2026-02-08 to 2026-03-27 for the second OSF version of the preprint.

## Repository structure

```
beliefs-narratives-networks/
  OtreeAnalysis/     # Analysis pipeline (data processing, figures, topic models)
  otreesurvey/       # oTree survey instrument (LLM-adaptive interviews, belief mapping)
  demo/              # demo from February 4 2026.(Video snippets and screenshots)
```

See the README in each subdirectory for details:

- [OtreeAnalysis/README.md](OtreeAnalysis/README.md) — Analysis pipeline: scripts, data flow, and figure generation
- [otreesurvey/README.md](otreesurvey/README.md) — Survey instrument: setup, deployment, and environment variables

The survey instrument uses a separate Whisper transcription server for voice input: [voice-whisper-server](https://github.com/victor-m-p/voice-whisper-server).
