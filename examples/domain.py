from dataclasses import dataclass


@dataclass
class MeshConfig:
    num_x: int
    x_l: float
    x_r: float


class Mesh:

    def __init__(self, **kwargs):
        if 'x' in kwargs:
            x = kwargs.get('x')
            self.x = x
            self.num_x = len(x)
            self.dx = x[1] - x[0]
        elif 'mesh_config' in kwargs:
            mesh_config = kwargs.get('mesh_config')
            self.num_x = mesh_config.num_x
            self.dx = (mesh_config.x_r - mesh_config.x_l) / (self.num_x - 1)
            from numpy import linspace
            if 'periodic' in kwargs and kwargs.get('periodic'):
                self.x = linspace(
                    mesh_config.x_l, mesh_config.x_r, self.num_x,
                    endpoint=False
                )
            else:
                self.x = linspace(
                    mesh_config.x_l, mesh_config.x_r, self.num_x
                )
            return
        else:
            raise ValueError('Must provide a MeshConfig object or '
                             'the mesh itself.')


class Operators:

    def __init__(self, mesh: Mesh, usr_np):
        self.mesh = mesh
        self.usr_np = usr_np

    def filt(self, f):
        return self.usr_np.concatenate((
            # i = 0
            [
                0.5 * (f[0] + f[1])
            ],
            # i = 1
            [
                0.5 * f[1] + 0.25 * (f[0] + f[2])
            ],
            # i = 2
            [
                (5/8) * f[2] + (1/4) * (f[1] + f[3]) -
                (1/16) * (f[0] + f[4])
            ],
            # i = 3
            [
                (11/16) * f[3] + (15/64) * (f[2] + f[4]) -
                (3/32) * (f[1] + f[5]) + (1/64) * (f[0] + f[6])
            ],
            # i = 4:-4
            (
                (93/128) * f[4:-4] +
                (7/32) * (f[3:-5] + f[5:-3]) -
                (7/64) * (f[2:-6] + f[6:-2]) +
                (1/32) * (f[1:-7] + f[7:-1]) -
                (1/256) * (f[:-8] + f[8:])
            ),
            # i = -4
            [
                (11/16) * f[-4] + (15/64) * (f[-3] + f[-5]) -
                (3/32) * (f[-2] + f[-6]) + (1/64) * (f[-1] + f[-7])
            ],
            # i = -3
            [
                (5/8) * f[-3] + (1/4) * (f[-2] + f[-4]) -
                (1/16) * (f[-1] + f[-5])
            ],
            # i = -2
            [
                0.5 * f[-2] + 0.25 * (f[-1] + f[-3])
            ],
            # i = -1
            [
                0.5 * (f[-1] + f[-2])
            ],
        ))

    def d_dx(self, f):
        return self.usr_np.concatenate((
            # i = 0
            [
                -90 * f[0] + 120 * f[1] - 30 * f[2]
            ],
            # i = 1
            [
                -90 * f[1] + 120 * f[2] - 30 * f[3]
            ],
            # i =2:-2
            (
                5 * f[:-4] - 40 * f[1:-3] + 0 * f[2:-2] +
                40 * f[3:-1] - 5 * f[4:]
            ),
            # i = -2
            [
                30 * f[-4] - 120 * f[-3] + 90 * f[-2]
            ],
            # i = -1
            [
                30 * f[-3] - 120 * f[-2] + 90 * f[-1]
            ]
        )) / (60 * self.mesh.dx)

    def laplacian(self, f):
        return self.usr_np.concatenate((
            [2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]],
            f[2:] - 2 * f[1:-1] + f[:-2],
            [-f[-4] + 4 * f[-3] - 5 * f[-2] + 2 * f[-1]]
        ))
