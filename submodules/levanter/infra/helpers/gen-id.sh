#!/bin/bash

# on osx tr has trouble reading /dev/urandom
TR=$(which gtr)

if [[ $? -ne 0 ]]; then
  TR=tr
fi

cat /dev/urandom | $TR -C -d a-z0-9 | head -c 8
