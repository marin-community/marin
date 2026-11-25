#!/usr/bin/env bash

while IFS= read -r -d '' assignment; do
  case "$assignment" in
    PATH=*) export "$assignment${PATH:+:$PATH}" ;;
    LD_LIBRARY_PATH=*) export "$assignment${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    *) export "$assignment" ;;
  esac
done < "$1"
unset assignment
