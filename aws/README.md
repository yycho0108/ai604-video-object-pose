# AWS Helper Scripts

## Install AWS CLI

```bash
./aws-cli-install.sh
```

## Run AWS Instance

```bash
./aws-run.sh
ssh -i ./current/top.pem "ubuntu@$(cat ./current/ip.txt)"
```

## Run Docker Container within AWS Instance

```bash
git clone https://github.com/yycho0108/ai604-video-object-pose.git
cd ai604-video-object-pose
docker build -t top
docker run -p 6006:6006 --volume /data:/home/user --shm-size 32G -it top
```


# Terminate AWS Instance

```bash
./aws-stop.sh
```
