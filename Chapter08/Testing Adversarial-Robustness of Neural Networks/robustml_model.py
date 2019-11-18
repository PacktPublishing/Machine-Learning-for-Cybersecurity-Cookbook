#!/usr/bin/env python3
import robustml
import numpy as np
import foolbox_model


class ABSModel(robustml.model.Model):
    """RobustML interface for the Analysis by Synthesis (ABS) model."""

    def __init__(self):
        self._dataset = robustml.dataset.MNIST()
        self._threat_model = robustml.threat_model.L2(epsilon=1.5)
        self._fmodel = foolbox_model.create()
        assert self._fmodel.bounds() == (0, 1)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        assert x.shape == (28, 28)
        x = x[np.newaxis]  # add chanell axis
        assert x.shape == (1, 28, 28)
        return np.argmax(self._fmodel.predictions(x))


if __name__ == '__main__':
    model = ABSModel()

    # design an input that looks like a 1
    x = np.zeros((28, 28), dtype=np.float32)
    x[5:-5, 12:-12] = 1

    print('class', model.classify(x))
