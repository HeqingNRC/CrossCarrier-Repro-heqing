#!/usr/bin/env bash
# Read-only resource discovery. Safe to run on the login node after SSH login.
set -uo pipefail

printf 'UTC: %s\nHost: %s\nUser: %s\nDirectory: %s\n' \
  "$(date -u +'%FT%TZ')" "$(hostname)" "${USER:-unknown}" "$PWD"
printf '\nSlurm partitions: name, availability, maximum time, generic resources\n'
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o '%P %a %l %G'
  scontrol show partition
else
  printf 'Slurm commands not found in the current shell.\n'
fi

printf '\nAllocation associations (may be restricted by site permissions)\n'
if command -v sacctmgr >/dev/null 2>&1; then
  sacctmgr -nP show assoc where user="${USER:-}" format=Account,Partition,QOS || true
fi

printf '\nAvailable environment modules\n'
if type module >/dev/null 2>&1; then
  module avail 2>&1 || true
else
  printf 'module command not available in this shell.\n'
fi

printf '\nCurrent Python, if present\n'
command -v python || true
python --version 2>&1 || true
printf '\nLinux C library (official recent wheels may require glibc >= 2.28)\n'
getconf GNU_LIBC_VERSION 2>&1 || true
printf '\nChoose account/partition/paths/module versions from your actual allocation.\n'
printf 'This probe does not allocate a GPU or run training.\n'
