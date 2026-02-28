#!/bin/bash
# Chạy LakeFlow: Qdrant + Backend + Frontend
# Usage: ./run.sh        (xem log)
#        ./run.sh -d     (chạy nền)
cd "$(dirname "$0")"
docker compose up "$@"
