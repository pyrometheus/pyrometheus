import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from pyroflow.boundary_conditions import BCConfig


@dataclass
class DomainConfig:
    num_x: int
    x_l: float
    x_r: float
    dtype: np.dtype = np.float64
    pyro_np = np
    num_dim: int = 1


class Domain:

    def __init__(self, domain_config: DomainConfig):
        self.dtype = domain_config.dtype
        self.num_dim = domain_config.num_dim
        self.num_x = domain_config.num_x
        self.pyro_np = domain_config.pyro_np

        self.dx = (domain_config.x_r - domain_config.x_l) / (self.num_x - 1)
        self.x = self.pyro_np.linspace(
            domain_config.x_l,
            domain_config.x_r,
            self.num_x,
            endpoint=True,
        )
        self.jac = self.pyro_np.array([self.dx])
