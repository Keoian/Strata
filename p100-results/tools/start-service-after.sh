#!/bin/bash
while kill -0 4195 2>/dev/null; do sleep 10; done
sleep 10
sudo systemctl start strata-bitlamas.service
echo "started $(date +%T): $(systemctl is-active strata-bitlamas.service)"
