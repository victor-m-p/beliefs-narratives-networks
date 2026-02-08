# otreesurvey

oTree v6 survey instrument for the beliefs-narratives-networks project. Implements LLM-adaptive semi-structured interviews, belief network elicitation, and pairwise comparison tasks. Deployed on Heroku with Prolific recruitment.

## Setup

Before running or deploying, the following environment variables must be set (in `.env` or your hosting platform):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (used for LLM-adaptive interviews) |
| `OTREE_ADMIN_PASSWORD` | oTree admin panel password |
| `OTREE_SECRET_KEY` | Django/oTree secret key for session signing |
| `PROLIFIC_COMPLETION_URL` | Prolific redirect URL for completed participants |
| `PROLIFIC_RETURN_URL` | Prolific redirect URL for returned participants |
| `PROLIFIC_NOCONSENT_URL` | Prolific redirect URL for non-consenting participants |

### Whisper transcription endpoint

The interview pages support voice input via a speech-to-text service. The endpoint URL is set as `WHISPER_TRANSCRIPTION_ENDPOINT` in the following templates and must be replaced with your own deployment URL before use:

- `InterviewMain.html`
- `InterviewTest.html`
- `InterviewFeedback.html`
- `PairInterviewLLM.html`
- `PairInterviewOpen.html`

## Deployment

1. Update `wave` in `settings.py`
2. Run `otree zip` (produces `otreesurvey.otreezip`)
3. Upload to oTreeHub and reset DB
4. Run `bash pilot_up.sh` to scale dynos

After data collection: run `bash pilot_down.sh` to scale down.

## Monitoring

- `heroku logs -a belief-narratives --tail`
- `heroku logs -a belief-narratives -n 2000 > heroku-logs.txt`
- `heroku addons -a belief-narratives` (Postgres)
- `heroku ps:type -a belief-narratives` (dynos)
