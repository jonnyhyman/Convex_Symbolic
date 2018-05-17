__version__ = '0.0'

from .generate import Generate
from .symbolic import Variable, Parameter, Constant
from .problem  import Minimize, Problem
from .canonicalize import Canonicalize

from .operations.functions.quads import (square, quad_over_lin,
                                                    sum_squares, inv_pos)
from .operations.functions.norms import norm

from .operations.functions.vectors import sum, maximum, minimum
from .operations.functions.radical import geo_mean, sqrt
from .operations.functions.scalars import abs

from .constraints import eq, ge, le

from .utilities import reshape
from .solution import solve
