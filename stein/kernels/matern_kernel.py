import numpy as np
from .abstract_kernel import AbstractKernel


class MaternKernel(AbstractKernel):
    """Matern-5/2 Kernel Class"""
    def kernel_and_grad(self, theta):
        """Implementation of abstract base class method."""
        raise NotImplementedError()
