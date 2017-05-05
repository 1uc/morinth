#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
#      Author : Luc Grosheintz <forbugrep@zoho.com>
#     Created : 2015-01-02
#
# Description : Push sources to hulk
read -r -d '' help_text << EOF
usage: $(basename $0) (push | pull) [--commit] [--remote <host>]
    Transefer data to or from a remote remote.

    pull    pull data in 'img'.
    push    copy source to the remote site

    --remote <remote> specify the remote location to copy to/from
EOF

if [[ $# < 1 ]]
then
  echo "$help_text"
  exit -1
fi

if [[ $# -ge 1 ]]
then
  case $1 in
    push|pull)
      mode="$1"
      ;;
    --help)
      echo "$help_text"
      exit 0
      ;;
    *)
      echo "You need to specify a mode, either 'push' or 'pull'."
      exit -1
      ;;
  esac

  shift
fi

# read-in the other for push
while [[ $# -ge 1 ]]
do
  case $1 in
    --remote)
      shift
      remote="$1"
      ;;

    *)
      echo "$help_text"
      exit -1
      ;;
  esac

  shift
done

## Configuration section.

# want to only test? (uncomment)
# DRY_RUN=--dry-run

# Set destination for each location
case $remote in
  daint)
    dest='lucg@ela.cscs.ch:~'
    ;;

  euler)
    dest='lucg@euler.ethz.ch:~'
    ;;

  ada)
    dest='lucg@ada-11:~'
    ;;

  *)
    echo "Unknown remote. Only know 'daint' and 'euler'."
    exit -1
    ;;
esac

## Actual copying up and down
case $mode in
  push)
    rsync -ur --delete $DRY_RUN \
          --include-from rsync.include --exclude-from rsync.exclude \
          $(realpath $PWD) $dest
    ;;

  pull)
    rsync -ruv $DRY_RUN "$dest/$(basename $(realpath $PWD))/img/*" img
    rsync -ruv $DRY_RUN "$dest/$(basename $(realpath $PWD))/data/*" data
    ;;

  *)
    echo "Unknown mode."
    exit -1
    ;;
esac
