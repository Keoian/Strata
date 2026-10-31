#!/bin/bash
# mon.sh OUT: every 20 s, RAM available + per-GPU memory/temp/power, until killed
while true; do echo "$(date +%T) $(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')MiB $(nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv,noheader | tr '\n' ' ')"; sleep 20; done >> $1
