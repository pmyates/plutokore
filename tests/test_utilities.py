import pytest
import os
from plutokore import utilities

def test_suppress_stdout():
    with utilities.suppress_stdout():
        print('This does not show')


def test_print_md():
    utilities.printmd('test')

