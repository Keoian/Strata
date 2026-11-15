#!/bin/sh
# the engine with --vision appended (tools/quant_quality.py drops --vision from a config's args)
exec /home/user/projects/strata/build-p100/strata "$@" --vision
