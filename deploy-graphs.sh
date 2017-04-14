#!/bin/bash

scp -r graphs urbica@stuurman.urbica.co:/mnt/storage/urbica/stuurman/
ssh urbica@stuurman.urbica.co 'bash -s' < deploy.sh
echo 'Done'
