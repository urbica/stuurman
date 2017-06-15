#!/bin/bash
set -e

cd /mnt/storage/urbica/stuurman
git pull
docker-compose up -d
