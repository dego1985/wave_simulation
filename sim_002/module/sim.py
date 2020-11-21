import numpy as np
import torch
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable

###############################
###   torch calculation   #####
def Padding(f, pad_sign=(1,1,1)):
    pad_ = ()
    for p in pad_sign:
        pad_ = (abs(p), abs(p)) + pad_
    f = F.pad(f, pad=pad_, mode='circular')
    if pad_sign[0] < 0:
        f[...,:-pad_sign[0],:,:] *= -1
        f[...,pad_sign[0]:,:,:] *= -1
    if pad_sign[1] < 0:
        f[...,:-pad_sign[1],:] *= -1
        f[...,pad_sign[1]:,:] *= -1
    if pad_sign[2] < 0:
        f[...,:-pad_sign[2]] *= -1
        f[...,pad_sign[2]:] *= -1
    return f


def der(f, dx=1, dim=0, sign=1):
    kernel = torch.cuda.FloatTensor(
        [-1, 0, 1]
    )
    dim_ = [1, 1, 1]
    dim_[dim] = 3
    kernel = kernel.expand(1, 1, dim_[0],  dim_[1],  dim_[2])
    pad_sign = [0, 0, 0]
    pad_sign[dim] = sign
    f = Padding(f, pad_sign=pad_sign)
    df = F.conv3d(f, kernel) 
    return df / (2 * dx)


def nabla2(f, dx=1, pad_sign=(1,1,1),scale=(1,1,1)):
    wx, wy, wz = scale
    kernel = torch.cuda.FloatTensor(
        [
        [[0, 0, 0],
         [0, wx, 0],
         [0, 0, 0]],
        [[0, wy, 0],
         [wz, -2*(wx + wy + wz), wz],
         [0, wy, 0]],
        [[0, 0, 0],
         [0, wx, 0],
         [0, 0, 0]],
        ]
    )
    kernel = kernel.expand(1, 1, 3, 3, 3)
    f = Padding(f, pad_sign=pad_sign)
    df = F.conv3d(f, kernel) 
    return df/(dx**2)


def gauss(f, pad_sign=(1,1,1)):
    kernel = torch.cuda.FloatTensor(
        [
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 6, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        ]
    )
    kernel = kernel.expand(1, 1, 3, 3, 3) / 12
    f = Padding(f, pad_sign=pad_sign)
    df = F.conv3d(f, kernel) 
    return df

class wave():
    def __init__(self, N=(100, 100, 100), dx=None, z_boundary=-1):
        self.N = N
        self.z_boundary = z_boundary
        if dx is None:
            max_N = max(N)
            dx = 2 * np.pi / max_N
        self.dx = dx
        X = [dx * (np.arange(N[i]) - N[i] / 2) for i in range(3)]
        x, y, z = X[0], X[1], X[2]
        x, y, z = np.meshgrid(x, y, z)

        w = torch.zeros([1, 1] + list(N)).cuda()
        v = torch.zeros([1, 1] + list(N)).cuda()
        t = torch.zeros([1]).cuda()

        # init value
        n = 10
        lx, ly, lz = dx * np.array(N)
        for _ in range(n):
            for Func in [np.cos, np.sin]:
                for xx in [v, w]:
                    a = (np.random.randint(4)-2) * 2 * np.pi/lx
                    b = (np.random.randint(4)-2) * 2 * np.pi/ly
                    c = (2*np.random.randint(5) + 1) * np.pi / lz
                    d = np.random.randn()
                    dv = d * torch.from_numpy(
                        Func(a * x + b * y + c * z).astype(np.float32)).cuda()
                    xx[:] = xx + dv/n

        self.w = w
        self.v = v
        self.t = t

    def update(self, dt=0):
        w = self.w
        v = self.v
        t = self.t
        dx = self.dx
        max_dt = 0.02
        scale = (1, 1, 1)

        step = int(dt / max_dt) + 1
        dt = dt / step
        # regist = torch.exp(-t/30)
        for _ in range(step):
            w = w + dt * v
            D2w = torch.tanh(nabla2(w, dx, pad_sign=(1,1,self.z_boundary), scale=scale))
            v = v + dt * D2w \
                + 20.0 * dt * w * (0.2 - w * w) \
                -0.0 * dt * v
                # - regist * 0.1 * dt * v
        alpha = 0.1
        w = (1 - alpha) * w + alpha * gauss(w, pad_sign=(1,1,self.z_boundary))

        self.w = w
        self.v = v
        self.t += dt

        return self.Z(w).squeeze()

    def update_numpy(self, dt=0):
        Z = self.update()
        Z = Z.to('cpu').detach().numpy().copy()

        return Z

    def Z(self, w):
        dim = 2
        N = self.N[dim]
        dx = self.dx

        # return w[0,0,:,:,10]

        # max_dw_z = torch.argmax(w, dim=-1) * dx  - N * dx / 2

        # return max_dw_z

        dw = gauss(w, pad_sign=(1,1,self.z_boundary))
        dw = der(w, dx=dx, dim=dim, sign=-1)
        dw = torch.abs(dw)
        max_dw_z = torch.argmax(dw, dim=-1) * dx - N * dx / 2

        return max_dw_z

        # dw = der(w, dx=dx, dim=dim, sign=-1)
        # dw = torch.abs(dw)
        # max_dw_z = torch.max(dw, dim=-1).values
        # min_dw_z = torch.min(dw, dim=-1).values

        # return (max_dw_z - min_dw_z) * 0.01
