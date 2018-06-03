from .setup_parameters import parameters
from . import setup_parameters

setup_parameters.setup_general_parameters()

from collections import namedtuple
Patient = namedtuple('Patient', ['geometry', 'data'])

from . import utils
from . import dolfin_utils
from . import io_utils
from . import numpy_mpi
from . import kinematics
from . import mechanicsproblem
from . import iterate
from . import unloader

# Subpackages
from . import material
from .material import *



from .unloader import (MeshUnloader, RaghavanUnloader,
                       FixedPointUnloader, HybridUnloader)
from .geometry import (Geometry, Marker, CRLBasis, HeartGeometry,
                       Microstructure, MarkerFunctions)
from .example_meshes import mesh_paths
from pulse.mechanicsproblem import (MechanicsProblem,
                                    BoundaryConditions,
                                    NeumannBC, RobinBC)
from .dolfin_utils import QuadratureSpace


# from .utils import logger
# from .dolfin_utils import RegionalParameter

from .kinematics import (SecondOrderIdentity,
                         DeformationGradient,
                         Jacobian,
                         GreenLagrangeStrain,
                         LeftCauchyGreen,
                         RightCauchyGreen,
                         EulerAlmansiStrain,
                         Invariants,
                         PiolaTransform,
                         InversePiolaTransform)

__version__ = '0.1'
__author__ = 'Henrik Finsberg'
__credits__ = ['Henrik Finsberg']
__license__ = 'LGPL-3'
__maintainer__ = 'Henrik Finsberg'
__email__ = 'henriknf@simula.no'
