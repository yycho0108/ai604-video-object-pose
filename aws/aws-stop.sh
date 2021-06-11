#!/usr/bin/env bash

set -ex

# Figure out script directory
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MYDIR="$(readlink -f ${MYDIR})"
MYDIR="${MYDIR}/current/"

aws ec2 terminate-instances --instance-ids "$(cat ${MYDIR}/instance.txt)"
