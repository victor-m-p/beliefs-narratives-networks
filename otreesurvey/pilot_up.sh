#!/usr/bin/env bash
set -euo pipefail

# Usage: ./pilot_up.sh [app-name] [dyno-size]
# Example: ./pilot_up.sh belief-narratives standard-1x
APP="${1:-belief-narratives}"
SIZE="${2:-standard-1x}"

echo "== Preparing $APP for pilot =="

# 0) Quick sanity: show current release & formation
heroku releases:info -a "$APP" || true
heroku ps -a "$APP" || true

# 1) Procfile should have ONLY the two oTree servers:
#    web:    otree prodserver1of2
#    worker: otree prodserver2of2
echo "== Remote Procfile =="
heroku run -a "$APP" 'bash -lc "test -f Procfile && cat Procfile || echo NO Procfile found"' || true
echo "If you still see a line that begins with 'rq:', please remove it from Procfile."

# 2) Ensure runtime metrics (idempotent)
heroku labs:enable log-runtime-metrics -a "$APP" || true

# 3) Put dyno types on the chosen size (ONLY web + worker)
heroku ps:type "web=$SIZE" "worker=$SIZE" -a "$APP"

# 4) Run one of each oTree server (NO rq)
heroku ps:scale web=1 worker=1 -a "$APP"

# 5) Show resulting formation & Postgres headroom
heroku ps -a "$APP"
heroku pg:info -a "$APP"

# 6) Friendly reminder if a Redis URL still lingers
if heroku config -a "$APP" | grep -q '^REDIS_URL='; then
  echo "⚠️  Found REDIS_URL in config. You can remove it with:"
  echo "    heroku config:unset REDIS_URL -a $APP"
fi
