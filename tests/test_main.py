import pytest

from word_vector_visualizer import main as target


def test_hello():
    assert target.hello() == "hello"
