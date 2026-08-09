#!/usr/bin/env bash
# Build the Milestone 1 deliverable PDF from its Markdown source.
#
# Renders docs/ML_PROBLEM_FRAMING_submission.md -> a print PDF with a running
# header ("MLOps LT 3 | Repository: aim-ml-ops"), page numbers, and
# keep-heading-with-content rules.
#
# Requires: pandoc, uv (for a throwaway weasyprint), and homebrew pango/cairo.
#   brew install pandoc pango
set -euo pipefail

cd "$(dirname "$0")/.."

SRC="docs/ML_PROBLEM_FRAMING_submission.md"
OUT="docs/MLOps LT3 Milestone 1 Deliverable.pdf"
CSS="scripts/deliverable.css"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# macOS: weasyprint loads pango via ctypes; point it at homebrew libs.
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-}:/opt/homebrew/lib"

pandoc "$SRC" -f gfm -t html5 -o "$WORK/body.html"

{
  printf '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><style>\n'
  cat "$CSS"
  printf '\n</style></head><body>\n'
  cat "$WORK/body.html"
  printf '\n</body></html>\n'
} > "$WORK/full.html"

uvx --with weasyprint python3 -c "
from weasyprint import HTML
HTML('$WORK/full.html').write_pdf('$OUT')
"

echo "Wrote: $OUT"
