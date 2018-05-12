import doctest

from kenchi.datasets import base


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(base))

    return tests
