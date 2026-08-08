#!/bin/bash
set -e

NOTEBOOKS_DIR=/app/notebooks
TEMPLATES_DIR=/app/templates
SOLUTIONS_DIR=/app/solutions

mkdir -p "$NOTEBOOKS_DIR"
mkdir -p "$NOTEBOOKS_DIR/_original_templates"

# Blank templates: always overwrite (reset on every start)
echo "📋 Resetting blank notebooks..."
for f in "$TEMPLATES_DIR"/*.ipynb; do
    [ -f "$f" ] || continue
    cp -f "$f" "$NOTEBOOKS_DIR/$(basename "$f")"
    cp -f "$f" "$NOTEBOOKS_DIR/_original_templates/$(basename "$f")"
done

# Solutions: always overwrite (kept in sync with image)
echo "📖 Syncing solution notebooks..."
for f in "$SOLUTIONS_DIR"/*.ipynb; do
    [ -f "$f" ] || continue
    cp -f "$f" "$NOTEBOOKS_DIR/$(basename "$f")"
done

echo "✅ Notebooks ready — launching JupyterLab"
exec jupyter lab \
    --ip=0.0.0.0 \
    --port="${PORT:-8888}" \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.tornado_settings='{"headers": {"Content-Security-Policy": "frame-ancestors *; style-src '"'"'self'"'"' '"'"'unsafe-inline'"'"' https://fonts.googleapis.com; font-src '"'"'self'"'"' https://fonts.gstatic.com"}}' \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True \
    --LabApp.news_url='' \
    --LabApp.default_url='/doc/tree/00_welcome.ipynb' \
    --notebook-dir="$NOTEBOOKS_DIR"
