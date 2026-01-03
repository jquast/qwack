#!/bin/sh
python tools/pregen_tiles.py --workers 16 --batch-size 10000 --default-tileset --default-charset --default-sizes --no-confusion
