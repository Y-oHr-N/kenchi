import doctest

from kenchi.datasets import samples_generator


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(samples_generator))

    return tests
