#!/usr/bin/env python3
import foolbox
import numpy as np
from abs_models.models import get_ABS
import os

def create():
    model = get_ABS()
    model.eval()
    fmodel = foolbox.models.PyTorchModel(
        model, (0, 1), num_classes=10)
    return fmodel


if __name__ == '__main__':
    fmodel = create()

    # design an input that looks like a 1
    x = np.zeros((1, 28, 28), dtype=np.float32)
    x[0, 5:-5, 12:-12] = 1

    logits = fmodel.predictions(x)

    print('logits', logits)
    print('probabilities', foolbox.utils.softmax(logits))
    print('class', np.argmax(logits))
