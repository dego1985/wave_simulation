import numpy as np
import torch
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable

###############################
###   torch calculation   #####
def Padding_2d(f, pad_sign=(1,1)):
    pad_ = ()
    for p in pad_sign:
        pad_ = (abs(p), abs(p)) + pad_
    f = F.pad(f, pad=pad_, mode='circular')
    if pad_sign[0] < 0:
        f[...,:-pad_sign[0],:] *= -1
        f[...,pad_sign[0]:,:] *= -1
    if pad_sign[1] < 0:
        f[...,:-pad_sign[1]] *= -1
        f[...,pad_sign[1]:] *= -1
    return f

def der_2d(f, dx=1, dim=0, sign=1):
    kernel = torch.cuda.FloatTensor(
        [-1, 0, 1]
    )
    dim_ = [1, 1, 1]
    dim_[dim] = 3
    kernel = kernel.expand(1, 1, dim_[0],  dim_[1])
    pad_sign = [0, 0]
    pad_sign[dim] = sign
    f = Padding_2d(f, pad_sign=pad_sign)
    df = F.conv2d(f, kernel) 
    return df / (2 * dx)


def nabla2_2d(f, dx=1, cyclic_length=1.0, sigma=0.1):
    kernels = torch.cuda.FloatTensor(
        [
        [[0, 1, 0],
         [0, -1, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, -1, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, -1, 0],
         [0, 1, 0]],
        [[0, 0, 0],
         [1, -1, 0],
         [0, 0, 0]],
        ]
    )
    dfs = []
    for kernel in kernels:
        kernel = kernel.expand(1, 1, 3, 3)
        f_pad = Padding_2d(f, pad_sign=(1,1))
        df = F.conv2d(f_pad, kernel)
        df = torch.sin(df * (2 * np.pi / cyclic_length))

        dfs.append(df)
    dfs = torch.cat(dfs, 0)
    dfs = torch.exp(- dfs**2 / (sigma**2)) * dfs
    df = dfs.sum(dim=0)
    f = df / dx
    return f


def gauss_2d_cb(f, cyclic_length=1.0, sigma=0.1):
    kernels = torch.cuda.FloatTensor(
        [
        [[0, 1, 0],
         [0, -1, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, -1, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, -1, 0],
         [0, 1, 0]],
        [[0, 0, 0],
         [1, -1, 0],
         [0, 0, 0]],
        ]
    )
    dfs = []
    for kernel in kernels:
        kernel = kernel.expand(1, 1, 3, 3)
        f_pad = Padding_2d(f, pad_sign=(1,1))
        df = F.conv2d(f_pad, kernel)
        df = torch.sin(df * (2 * np.pi / cyclic_length))

        dfs.append(df)
    dfs = torch.cat(dfs, 0)
    dfs = torch.exp(- dfs**2 / (sigma**2)) * dfs
    df = dfs.sum(dim=0)
    f = f + df / 8
    return f

class wave():
    def __init__(self, N=(100, 100), dx=1.0, cyclic_length=1, sigma=0.1):
        self.N = N
        self.cyclic_length = cyclic_length
        self.sigma = sigma
        if dx is None:
            max_N = max(N)
            dx = 2 * np.pi / max_N
        self.dx = dx
        X = [dx * (np.arange(N[i]) - N[i] / 2) for i in range(2)]
        x, y = X[0], X[1]
        x, y = np.meshgrid(x, y)

        z = torch.zeros([1, 1] + list(N)).cuda()
        v = torch.zeros([1, 1] + list(N)).cuda()
        t = torch.zeros([1]).cuda()

        # init value
        n = 10
        NN = 10
        lx, ly = dx * np.array(N)
        for _ in range(n):
            for Func in [np.cos, np.sin]:
                for xx in [v, z]:
                    a = (np.random.randint(NN)-NN//2) * 2 * np.pi/lx
                    b = (np.random.randint(NN)-NN//2) * 2 * np.pi/ly
                    d = np.random.randn()
                    dv = d * torch.from_numpy(
                        Func(a * x + b * y).astype(np.float32)).cuda()
                    xx[:] = xx + dv/n*5

        self.z = z + 0.5
        self.v = v
        self.t = t

    def update(self, dt=0):
        z = self.z
        v = self.v
        dx = self.dx
        cyclic_length = self.cyclic_length
        sigma = self.sigma

        max_dt = 0.01

        step = int(dt / max_dt) + 1
        dt = dt / step
        for _ in range(step):
            z = z + dt * v
            D2z = nabla2_2d(z, dx, cyclic_length=cyclic_length, sigma=sigma)
            v = v + dt * D2z \
                -0.01 * dt * v

            # alpha = 0.1 * dt
            # z = (1 - alpha) * z + alpha * gauss_2d_cb(z, sigma=sigma)

        # z cyclic
        z = z - torch.floor(z/cyclic_length) * cyclic_length
        self.z = z
        self.v = v

        return self.z.squeeze()

    def update_numpy(self, dt=0):
        Z = self.update()
        Z = Z.to('cpu').detach().numpy().copy()

        return Z



