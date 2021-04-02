# ai604-video-object-pose

Video Object Pose estimation project for KAIST 2021SP AI604 : Computer Vision

## TFRecord Parsing

Requires a modified version of the `tfrecord` package to handle the following list of complications:

* Loading from GCS (Google Cloud Storage) bucket
* Loading a `SequenceExample`
* Loading from shard(s)

One way to install the package would be the following:

```bash
pip3 install git+https://github.com/yycho0108/tfrecord.git@gcs-seq
```

For further details, visit [yycho0108/tfrecord](https://github.com/yycho0108/tfrecord.git) instead.
