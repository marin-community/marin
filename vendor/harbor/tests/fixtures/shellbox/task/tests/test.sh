#!/bin/sh
mkdir -p /logs/verifier
if [ "$(cat /workspace/answer.txt)" = "hello from qemu" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
