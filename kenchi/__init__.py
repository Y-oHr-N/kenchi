# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/

__version__          = '0.10.0'

try:
    __KENCHI_SETUP__
except NameError:
    __KENCHI_SETUP__ = False

if not __KENCHI_SETUP__:
    from . import datasets # noqa
    from . import metrics # noqa
    from . import outlier_detection # noqa
    from . import pipeline # noqa
    from . import plotting # noqa
    from . import utils # noqa
