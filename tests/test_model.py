import time
from src import model

def test_inference_accuracy():
    inputs = [1, 2, 3]
    expected = [2, 4, 6]
    assert model.predict(inputs) == expected

def test_inference_time():
    inputs = list(range(10000))
    start = time.time()
    _ = model.predict(inputs)
    duration = time.time() - start
    assert duration < 1.0
