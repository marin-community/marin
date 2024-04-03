# vendors a few parts of guesslang so we can use it now that TF has deprecated the estimator API.
import json
import os
import timeit
from operator import itemgetter

import tensorflow as tf

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'data/model')
LANGUAGE_MAPPING_FILE = os.path.join(os.path.dirname(__file__), 'data/languages.json')


class Model(object):
    def __init__(self):
        self.model = load_model(MODEL_FILE)
        self.forward_mapping = load_language_mapping(LANGUAGE_MAPPING_FILE)
        self.mapping = {v: k for k, v in self.forward_mapping.items()}

    def predict(self, text):
        return predict(self.model, self.mapping, text)


def load_model(saved_model_dir: str):
    """Load a Tensorflow saved model"""
    return tf.saved_model.load(saved_model_dir)


def load_language_mapping(mapping_file: str) -> dict[str, str]:
    """Load the language mapping from a file"""
    with open(mapping_file, 'r') as f:
        return dict(json.load(f))





def predict(
    saved_model,
    mapping: dict[str, str],
    text: str
) -> list[tuple[str, float]]:
    """Infer a Tensorflow saved model"""
    content_tensor = tf.constant([text])
    predicted = saved_model.signatures['serving_default'](content_tensor)

    numpy_floats = predicted['scores'][0].numpy()
    extensions = predicted['classes'][0].numpy()

    probability_values = (float(value) for value in numpy_floats)
    languages = (mapping.get(ext.decode(), ext.decode()) for ext in extensions)

    unsorted_scores = zip(languages, probability_values)
    scores = sorted(unsorted_scores, key=itemgetter(1), reverse=True)
    return scores


if __name__ == '__main__':
    model = Model()
    # load this file and predict the language

    with open('guess_code.py', 'r') as f:
        code = f.read()
        t = timeit.timeit(lambda: model.predict(code), number=100)
        print(f'Average time: {t:.3f} seconds')

    print(model.predict(code))
