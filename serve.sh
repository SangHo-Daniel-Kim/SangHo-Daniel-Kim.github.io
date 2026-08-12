#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export GEM_HOME=/u/sajakim/.gems/ruby/3.2.0
export GEM_PATH=/u/sajakim/.gems/ruby/3.2.0
export PATH="/u/sajakim/.gems/ruby/3.2.0/bin:$PATH"

pkill -f "jekyll serve" 2>/dev/null || true
sleep 1

echo "Building site..."
jekyll build

echo "Starting server at http://0.0.0.0:4000 ..."
exec jekyll serve --livereload --port 4000 --host 0.0.0.0 --force_polling
