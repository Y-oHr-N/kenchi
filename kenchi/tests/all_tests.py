import unittest


def suite():
    return unittest.defaultTestLoader.discover('.', 'test_*.py')
