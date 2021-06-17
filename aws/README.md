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

## Configure Package Within Docker Container

First, install `tmux` for multitasking:
```
# tmux inside docker (TODO(ycho): add to Dockerfile)
sudo apt install tmux
```

BTW, To generate `requirements.txt` needed in later stages:
```bash
poetry export -f requirements.txt --dev --without-hashes --output /tmp/requirements.txt
```

Go through unnecessary lengths of hackery with the packaging ecosystem
in order to enable editable installs:

```bash
# Generate setup.py
pip3 install poetry2setup --user
poetry2setup > setup.py

# Workaround stupid bugs w.r.t conda, etc
pip3 install -r requirements.txt # TODO(ycho): needed?
python3 setup.py develop --install-dir ~/.local/lib/python3.6/site-packages
```

Check that everything is fine:

```bash
python3 -c 'import top; print(top.__path__)'
# ['/data/ai604-video-object-pose/src/top']
```

## Terminate AWS Instance

```bash
./aws-stop.sh
```
