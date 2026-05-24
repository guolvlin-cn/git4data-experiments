#!/usr/bin/env bash
# Start a single-node lakeFS server in Docker, backed by Alibaba OSS.
#
#   metadata KV : local (in-container) — no Postgres needed
#   blockstore  : s3 adapter pointed at OSS (S3-compatible endpoint)
#   API creds   : pre-seeded via LAKEFS_INSTALLATION_* so the client knows them
#
# Reads OSS + lakeFS settings from ../.lakefs.env
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .lakefs.env ] || { echo "missing .lakefs.env (copy .lakefs.env.example)"; exit 1; }
set -a; . ./.lakefs.env; set +a

docker rm -f lakefs >/dev/null 2>&1 || true

docker run -d --name lakefs -p 127.0.0.1:8200:8000 \
  -e LAKEFS_DATABASE_TYPE=local \
  -e LAKEFS_AUTH_ENCRYPT_SECRET_KEY=ml-git4data-demo-secret \
  -e LAKEFS_BLOCKSTORE_TYPE=s3 \
  -e LAKEFS_BLOCKSTORE_S3_ENDPOINT="${OSS_ENDPOINT}" \
  -e LAKEFS_BLOCKSTORE_S3_REGION="${OSS_REGION}" \
  -e LAKEFS_BLOCKSTORE_S3_FORCE_PATH_STYLE=false \
  -e LAKEFS_BLOCKSTORE_S3_CREDENTIALS_ACCESS_KEY_ID="${OSS_ACCESS_KEY_ID}" \
  -e LAKEFS_BLOCKSTORE_S3_CREDENTIALS_SECRET_ACCESS_KEY="${OSS_ACCESS_KEY_SECRET}" \
  -e LAKEFS_INSTALLATION_USER_NAME=admin \
  -e LAKEFS_INSTALLATION_ACCESS_KEY_ID="${LAKEFS_ACCESS_KEY_ID}" \
  -e LAKEFS_INSTALLATION_SECRET_ACCESS_KEY="${LAKEFS_SECRET_ACCESS_KEY}" \
  treeverse/lakefs:latest run

echo "waiting for lakeFS to become healthy ..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8200/_health >/dev/null 2>&1; then
    echo "lakeFS is up at http://127.0.0.1:8200  (login with the keys in .lakefs.env)"
    exit 0
  fi
  sleep 1
done
echo "lakeFS did not become healthy in time — check: docker logs lakefs"
exit 1
