#!/usr/bin/env python3
"""Plot Objectron evaluation results from evaluate_model.py.

Expects the report file to have been saved at "/tmp/report.txt".
Requires `ttp`, which may be installed as `pip3 install --user ttp`.
TODO(ycho): `poetry add ttp --dev`
"""

from ttp import ttp
import json
import numpy as np
from matplotlib import pyplot as plt


def main(filename='/tmp/report.txt'):
    template = R"""Mean Error 2D: {{MPE2D | to_float }}
Mean 3D IoU: {{MIOU | to_float }}
Mean Azimuth Error: {{MAE | to_float }}
Mean Polar Error: {{MPE | to_float }}

IoU Thresholds: {{iou_thresh | _line_ }}
AP @3D IoU     : {{ap_iou | _line_}}

2D Thresholds : {{pixel_thresh | _line_}}
AP @2D Pixel   : {{ap_pixel | _line_}}

Azimuth Thresh: {{azim_thresh | _line_}}
AP @Azimuth     : {{ap_azim | _line_}}

Polar Thresh   : {{polar_thresh | _line_}}
AP @Polar       : {{ap_polar | _line_}}

ADD Thresh     : {{add_thresh | _line_}}
AP @ADD             : {{ap_add | _line_}}

ADDS Thresh     : {{adds_thresh | _line_}}
AP @ADDS           : {{ap_adds | _line_}}"""
    with open(filename) as fp:
        data = fp.read()

    parser = ttp(data=data, template=template)
    parser.parse()
    # print(parser.vars)
    res = json.loads(parser.result(format='json')[0])[0]
    print(res)
    res = {
        k: np.fromstring(v, sep=',\t') if isinstance(
            v, str) else v for (
                k, v) in res.items()}
    print(res)
    # print(res)
    plt.plot(res['iou_thresh'], res['ap_iou'])
    plt.show()


if __name__ == '__main__':
    main()
