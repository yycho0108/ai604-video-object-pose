#!/usr/bin/env bash

# Figure out script directory
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MYDIR="$(readlink -f ${MYDIR})"

aws ec2 terminate-instances --instance-ids "$(cat ${MYDIR}/instance.txt)"
