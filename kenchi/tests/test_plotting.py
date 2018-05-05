import doctest

from kenchi import plotting


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(plotting))

    return tests
