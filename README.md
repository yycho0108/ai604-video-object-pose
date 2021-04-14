# TOP

Temporal Object Pose Estimation : KAIST 2021SP AI604


Video Object Pose estimation project for KAIST 2021SP AI604 : Computer Vision

## Installation/Development

### Installing dependencies

With `pip`:

```bash
pip3 install -r requirements.txt # --user
```

See [requirements.txt](requirements.txt) for this case.

Alternatively, with `poetry`:

```bash
poetry update
```

See [pyproject.toml](pyproject.toml) for this case which includes the specs for the compatibility matrix.

Note that as of **04-15-2021**, installation with `poetry` requires latest changes from master. This is due to the version specification conventions for `torch` and `torchvision` (i.e. the `+cuXXX` modifiers)
See [poetry#3831](https://github.com/python-poetry/poetry/pull/3831).


### TFRecord Parsing

Requires a modified version of the `tfrecord` package to handle the following list of complications:

* Loading from GCS (Google Cloud Storage) bucket
* Loading a `SequenceExample`
* Loading from shard(s)

One way to install the package would be the following:

```bash
pip3 install git+https://github.com/yycho0108/tfrecord.git@gcs-seq
```

For further details, visit [yycho0108/tfrecord](https://github.com/yycho0108/tfrecord.git) instead.

## Authors

Yoonyoung (Jamie) Cho : yoonyoung.cho@kaist.ac.kr
Jiyong Ahn : ajy8456@gmail.com
