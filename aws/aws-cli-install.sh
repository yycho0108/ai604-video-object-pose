#!/usr/bin/env bash

# Download installer
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"

# Install & remove
unzip /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install && rm /tmp/awscliv2.zip && rm -rf /tmp/aws
