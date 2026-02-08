#!/usr/bin/env bash
set -euo pipefail

APP="${1:-belief-narratives}"

# Turn off background dynos
heroku ps:scale worker=0 -a "$APP"

# Keep one cheap web dyno
heroku ps:type web=basic -a "$APP"

echo
echo "== Add-ons (pricing & plans) =="
heroku addons -a "$APP"

echo
echo "== Dyno types (pricing & plans) =="
heroku ps:type -a "$APP"

    """_summary_
    """echo "✅ Done."