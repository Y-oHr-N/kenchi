from setuptools import find_packages, setup

from kenchi import __version__


with open('README.rst') as f:
    readme           = f.read()

setup(
    name             = 'kenchi',
    version          = __version__,
    author           = 'Kon',
    author_email     = 'kon.y.ohr.n@gmail.com',
    url              = 'http://y-ohr-n.github.io/kenchi',
    description      = 'A set of python modules for anomaly detection',
    long_description = readme,
    license          = 'MIT',
    packages         = find_packages(exclude=['tests']),
    install_requires = ['numpy', 'scipy', 'scikit-learn'],
    test_suite       = 'kenchi.tests'
)
