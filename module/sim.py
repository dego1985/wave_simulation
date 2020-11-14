import numpy as np
import torch
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable

###############################
###   torch calculation   #####


def Dx(f, d=1):
    kernel = torch.cuda.FloatTensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]])
    kernel = kernel.expand(1, 1, 3, 3)/4
    f = F.pad(f, (1, 1, 1, 1), mode='circular')
    df = [F.conv2d(f[:, i:i+1, :, :], kernel) for i in range(f.shape[1])]
    df = torch.cat(df, 1)
    return df/d


def Dy(f, d=1):
    kernel = torch.cuda.FloatTensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]])
    kernel = kernel.expand(1, 1, 3, 3)/4
    f = F.pad(f, (1, 1, 1, 1), mode='circular')
    df = [F.conv2d(f[:, i:i+1, :, :], kernel) for i in range(f.shape[1])]
    df = torch.cat(df, 1)
    return df/d

def Dy2(f, d=1):
    # kernel = torch.cuda.FloatTensor(
    #     [[1, 2, 1],
    #      [-2, -4, -2],
    #      [1, 2, 1]])
    kernel = torch.cuda.FloatTensor(
        [[0, 4, 0],
         [0, -8, 0],
         [0, 4, 0]])
    kernel = kernel.expand(1, 1, 3, 3)/4
    f = F.pad(f, (1, 1, 1, 1), mode='circular')
    df = [F.conv2d(f[:, i:i+1, :, :], kernel) for i in range(f.shape[1])]
    df = torch.cat(df, 1)
    return df/(d**2)

def Dx2(f, d=1):
    kernel = torch.cuda.FloatTensor(
        [[0, 0, 0],
         [4, -8, 4],
         [0, 0, 0]])
    # kernel = torch.cuda.FloatTensor(
    #     [[1, -2, 1],
    #      [2, -4, 2],
    #      [1, -2, 1]])
    kernel = kernel.expand(1, 1, 3, 3)/4
    f = F.pad(f, (1, 1, 1, 1), mode='circular')
    df = [F.conv2d(f[:, i:i+1, :, :], kernel) for i in range(f.shape[1])]
    df = torch.cat(df, 1)
    return df/(d**2)

class wave():
    def __init__(self):
        self.N = N = 1000
        self.dx = 2 * np.pi / N
        x = np.arange(N) / N * 2 * np.pi - np.pi
        y = np.arange(N) / N * 2 * np.pi - np.pi
        x, y = np.meshgrid(x, y)

        z = torch.zeros([1, 1, N, N]).cuda()
        v = torch.zeros([1, 1, N, N]).cuda()

        n = 10
        for _ in range(n):
            a = np.random.randint(4)
            b = np.random.randint(4)
            c = np.random.randn()
            d = c * torch.from_numpy(
                np.sin(a * x + b * y).astype(np.float32)).cuda()
            v = v + d/n

            a = np.random.randint(4)
            b = np.random.randint(4)
            c = np.random.randn()
            d = c * torch.from_numpy(
                np.cos(a * x + b * y).astype(np.float32)).cuda()
            v = v + d/n

            a = np.random.randint(4)
            b = np.random.randint(4)
            c = np.random.randn()
            d = c * torch.from_numpy(
                np.sin(a * x + b * y).astype(np.float32)).cuda()
            z = z + d/n

            a = np.random.randint(4)
            b = np.random.randint(4)
            c = np.random.randn()
            d = c * torch.from_numpy(
                np.cos(a * x + b * y).astype(np.float32)).cuda()
            z = z + d/n

        self.z = z * 0.1
        self.v = v * 0.1

    def update(self, dt=0):
        z = self.z
        v = self.v
        dx = self.dx
        max_dt = 0.01

        N = int(dt / max_dt) + 1
        dt = dt / N
        for _ in range(N):
            z = z + dt * v
            D2z = torch.tanh(Dx2(z, dx) + Dy2(z, dx))
            v = v + dt * D2z \
                + 2.0 * dt * z * (0.1 - z * z) \
                - 0.01 * dt * v

        self.z = z
        self.v = v

        return z.squeeze()

    def update_numpy(self, dt=0):
        self.update()
        z = self.z
        z = z.to('cpu').detach().numpy().copy()
        return z.squeeze()
