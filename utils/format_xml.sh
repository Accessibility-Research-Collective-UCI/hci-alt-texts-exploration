#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-.}"

if ! command -v xmllint >/dev/null 2>&1; then
  echo "Error: xmllint is not installed." >&2
  exit 1
fi

if [[ ! -d "$DIR" ]]; then
  echo "Error: '$DIR' is not a directory." >&2
  exit 1
fi

failed=0
formatted=0

while IFS= read -r -d '' file; do
  tmp="$(mktemp "${TMPDIR:-/tmp}/xmllint-format.XXXXXX")"

  if xmllint --format "$file" > "$tmp"; then
    # Overwrite the existing file contents while preserving ownership/permissions.
    cat "$tmp" > "$file"
    ((formatted++))
    echo "Formatted: $file"
  else
    ((failed++))
    echo "Failed: $file" >&2
  fi

  rm -f "$tmp"
done < <(find "$DIR" -type f -name "*.xml" -print0)

echo "Done. Formatted: $formatted, Failed: $failed"

if (( failed > 0 )); then
  exit 1
fi