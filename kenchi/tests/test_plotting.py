import doctest

from kenchi import plotting
from sklearn.utils.testing import if_matplotlib


@if_matplotlib
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(plotting))

    return tests
