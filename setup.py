from setuptools import find_packages, setup

from kenchi import __version__


with open('README.rst') as f:
    readme           = f.read()

with open('requirements.txt') as f:
    requires         = f.read().splitlines()

setup(
    name             = 'kenchi',
    version          = __version__,
    author           = 'Kon',
    author_email     = 'kon.y.ohr.n@gmail.com',
    url              = 'http://kenchi.readthedocs.io',
    description      = 'A set of python modules for anomaly detection',
    long_description = readme,
    license          = 'MIT',
    packages         = find_packages(exclude=['tests']),
    install_requires = requires,
    test_suite       = 'kenchi.tests.suite'
)
