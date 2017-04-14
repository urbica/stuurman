#!/bin/bash
set -e

cd /mnt/storage/sergeygo/router
git pull
docker-compose build
docker-compose down
docker-compose up -d
