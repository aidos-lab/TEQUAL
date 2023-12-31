#!/usr/bin/env bash
# Driver Script to prevent python from doing weird memory shit.

if [ -f ../../.env ]; then
    source ../../.env
fi
if [ -n "$params" ]; then
EXPERIMENT="$(cat $params | shyaml get-value experiment)"
fi


export PYTORCH_ENABLE_MPS_FALLBACK=1
export CODECARBON_GPU_IDS="all"

# CONFIGS="$root/experiments/$EXPERIMENT/configs/*"
CONFIGS=`ls $root/experiments/$EXPERIMENT/configs/config_* | sort -V`

cd "$root/src"
for c in $CONFIGS
do
    poetry run python train.py -c $c
done