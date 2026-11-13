#!/bin/sh
# Score one split on a Kaggle T4:  kaggle/push.sh dev [limit]  |  kaggle/push.sh test
# The script is sent as it is in the working tree, which must be clean so that the commit
# recorded in run.json identifies it. Fetch the results afterwards with:
#   uvx --with six kaggle kernels output $KAGGLE_USER/right-sizing-<split> -p /tmp/out && cp -r /tmp/out/results .
set -eu
user=${KAGGLE_USER:-gonzalodazamor}
split=$1
limit=${2:-0}
cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain)" ] || { echo "commit your changes first"; exit 1; }
slug="right-sizing-$split"
dir=$(mktemp -d)
sed -e "s|@SPLIT@|$split|" -e "s|@LIMIT@|$limit|" -e "s|@COMMIT@|$(git rev-parse HEAD)|" kaggle/run.py > "$dir/run.py"
cat > "$dir/kernel-metadata.json" <<JSON
{"id": "$user/$slug", "title": "$slug", "code_file": "run.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "kernel_sources": [], "dataset_sources": [], "competition_sources": []}
JSON
uvx --with six kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4
echo "status: uvx --with six kaggle kernels status $user/$slug"
