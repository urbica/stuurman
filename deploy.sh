#!/bin/bash
set -e

cd /mnt/storage/urbica/stuurman
git pull
docker-compose build
docker-compose down
docker-compose up -d
