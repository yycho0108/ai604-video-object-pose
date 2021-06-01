#!/usr/bin/env bash

set -ex


# === CONFIGURE ===
PROJECT_NAME='top' # CONFIGURE PROJECT NAME
AWS_AMI='ami-04f97d9f48a04bb4a' # AWS Deep Learning AMI Ubuntu 18.04
AWS_REGION='ap-northeast-2' # Region Asia Pacific (Seoul)
AWS_INSTANCE_TYPE='p3.2xlarge' # Instance Type
AWS_USER="ubuntu" # FIXME(ycho): USER NAME DEPENDS ON AMI! `ec2-user` or `ubuntu`


# CHECK aws exists
if ! command -v aws &> /dev/null; then
    echo 'install aws with ./aws-cli-install.sh'
    exit
fi

# CHECK netcat exists
# FIXME(ycho): Maybe not *really* necessary to require netcat
if ! command -v nc.traditional &> /dev/null; then
    echo 'install netcat-traditional with apt-get install netcat-traditional'
    exit
fi

# Figure out my IP
HOST_IP="$(wget http://ipinfo.io/ip -qO -)"

# Figure out script directory
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MYDIR="$(readlink -f ${MYDIR})"
MYDIR="${MYDIR}/current/"
mkdir -p "${MYDIR}"

# Remove pre-existing credentials...
aws ec2 delete-key-pair --key-name "${PROJECT_NAME}" --region "${AWS_REGION}"
aws ec2 delete-security-group --group-name "${PROJECT_NAME}" || true
rm "${MYDIR}/${PROJECT_NAME}.pem" || true

# Create new credentials...
aws ec2 create-key-pair --key-name "${PROJECT_NAME}" --region "${AWS_REGION}" \
    --query 'KeyMaterial' --output text > "${MYDIR}/${PROJECT_NAME}.pem"
    chmod 600 "${MYDIR}/${PROJECT_NAME}.pem"
    aws ec2 create-security-group --group-name "${PROJECT_NAME}" --description "${PROJECT_NAME}"
    aws ec2 authorize-security-group-ingress --group-name "${PROJECT_NAME}" \
        --ip-permissions "[{\"IpProtocol\": \"tcp\", \"FromPort\": 0, \"ToPort\": 65535, \"IpRanges\": [{\"CidrIp\": \"$HOST_IP/24\"}]}]"

# Finally, launch!
AWS_LAUNCH_JSON=$(aws ec2 run-instances \
    --image-id "${AWS_AMI}" \
    --region "${AWS_REGION}" \
    --count 1 \
    --instance-type "${AWS_INSTANCE_TYPE}" \
    --key-name "${PROJECT_NAME}" \
    --security-groups "${PROJECT_NAME}" \
    --block-device-mappings 'file://block-device-mapping.json')

# SAVE LAUNCH CFG ...
echo "${AWS_LAUNCH_JSON}" > "${MYDIR}/aws-launch.json"

# == GET IP ADDRESS ==
AWS_IP=''
while [ -z "${AWS_IP}" ]; do
    AWS_IP="$(aws ec2 describe-instances --filter Name="key-name",Values="${PROJECT_NAME}" "Name=instance-state-name,Values=running" --query 'Reservations[].Instances[].[PublicIpAddress]' --output text)"
done
echo "${AWS_IP}" > "${MYDIR}/ip.txt"
# ======================

# Continue until the SSH connection is ready
SSH_READY=''
while [ -z ${SSH_READY} ]; do
    echo "Trying to connect..."
    if [ "$(nc.traditional $(aws ec2 describe-instances --filter Name='key-name',Values=\"${PROJECT_NAME}\" 'Name=instance-state-name,Values=running' --query 'Reservations[].Instances[].[PublicIpAddress]' --output text) -z -w 4 22; echo $?)" = 0 ]; then
        SSH_READY=true;
    fi
    sleep 1
done

# Poll SSH to get instance ID to ensure that SSH connection can be acquired
AWS_INSTANCE_ID=''
while [ -z "${AWS_INSTANCE_ID}" ]; do
    sleep 1
    AWS_INSTANCE_ID=$(ssh -vvv -i "${MYDIR}/${PROJECT_NAME}.pem" -o StrictHostKeyChecking=no "${AWS_USER}@${AWS_IP}" "curl -s http://169.254.169.254/latest/meta-data/instance-id; echo")
done
echo "${AWS_INSTANCE_ID}" > "${MYDIR}/instance.txt"

# Mount EBS volume in the remote ...
# FIXME(ycho): hardcoded `ubuntu` inside mounting script
ssh -tt -i "${MYDIR}/${PROJECT_NAME}.pem" -o StrictHostKeyChecking=no "${AWS_USER}@${AWS_IP}" -- \
    "sudo su root -c 'mkfs -t xfs /dev/xvdb && mkdir /data && chown ubuntu:ubuntu /data && mount /dev/xvdb /data'"

# Copy already configured AWS credentials from ${HOME}
# FIXME(ycho): NOT the best idea but works.
scp -r -i "${MYDIR}/${PROJECT_NAME}.pem" -o StrictHostKeyChecking=no -- \
    "${HOME}/.aws" "${AWS_USER}@${AWS_IP}:/home/${AWS_USER}"

# Configure Docker on the remote
ssh -tt -i "${MYDIR}/${PROJECT_NAME}.pem" -o StrictHostKeyChecking=no "${AWS_USER}@${AWS_IP}" -- \
    "aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com"

# Pull docker image on the remote
# FIXME(ycho): Super hardcoded!!!!!!!!!!!
ssh -tt -i "${MYDIR}/${PROJECT_NAME}.pem" -o StrictHostKeyChecking=no "${AWS_USER}@${AWS_IP}" -- \
    docker pull 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
