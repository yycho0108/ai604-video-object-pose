#!/usr/bin/env python3

from enum import Enum


class Topic(Enum):
    """List of pre-defined topics."""
    STEP = 'step'
    EPOCH = 'epoch'
    TRAIN_BEGIN = 'train_begin'  # Null-ary signal at beginning of training.
    TRAIN_LOSS = 'total_loss'  # Return total training loss
    TRAIN_LOSSES = 'train_losses'  # Return dictionary of [name, value]
    METRICS = 'metrics'  # Event that fired after all metrics are calculated.
    TRAIN_OUT = 'train_out'  # Dict of model outputs during training.

    EVAL_BEGIN = 'eval_begin'
    EVAL_STEP = 'eval_step'
    EVAL_END = 'eval_end'
