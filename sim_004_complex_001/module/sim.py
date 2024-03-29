import numpy as np
import torch
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable

###############################
###   torch calculation   #####

pi = torch.from_numpy(np.array(np.pi))

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


def conv2d_complex(f, kernel):

    f_re = f.real
    f_im = f.imag
    df_re = F.conv2d(f_re, kernel)
    df_im = F.conv2d(f_im, kernel)
    df = torch.complex(df_re, df_im)
    return df

def nabla2_2d(f, dx=1):
    kernel = torch.cuda.FloatTensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
    )
    kernel = kernel.expand(1, 1, 3, 3)
    f_pad = Padding_2d(f, pad_sign=(1,1))
    df = conv2d_complex(f_pad, kernel)
    return df / (dx**2)


class wave():
    def __init__(self, N=(100, 100), dx=1.0):
        self.N = N
        self.dx = dx

        X = [dx * (np.arange(N[i]) - N[i] / 2) for i in range(2)]
        x, y = X[0], X[1]
        x, y = np.meshgrid(x, y)

        z = np.zeros([1, 1] + list(N), dtype=np.complex64)
        v = np.zeros([1, 1] + list(N), dtype=np.complex64)

        # init value
        n = 10
        NN = 2
        lx, ly = dx * np.array(N)
        for _ in range(n):
            for xx in [v, z]:
                a = np.random.random_integers(-NN,NN) * (2 * np.pi) / lx
                b = np.random.random_integers(-NN,NN) * (2 * np.pi) / ly
                d = np.random.randn() + 1j* np.random.randn()
                dv = d * np.exp(1j*a * x + 1j*b * y)
                xx[:] = xx + dv/np.sqrt(n)

        # torch variables
        z = torch.from_numpy(z).cuda() * 0.4
        v = torch.from_numpy(v).cuda() * 0.4
        t = torch.zeros([1]).cuda()

        zs, mask = self.make_ele()

        self.z = z
        self.v = v
        self.t = t
        self.zs = zs
        self.mask = mask

    def update(self, dt=0):
        z = self.z
        v = self.v
        t = self.t
        dx = self.dx
        zs = self.zs
        mask = self.mask

        max_dt = 0.01

        step = int(dt / max_dt) + 1
        dt = dt / step
        for _ in range(step):
            z = z + dt * v
            D2z = nabla2_2d(z, dx)
            v = v + 0.05*dt * D2z \
                + 0.1 * dt * z * (1 - z.conj()*z)
            
            v += - 0.01 * dt * v
            # z, v = self.put(z, v, zs, mask)

        self.z = z
        self.v = v
        self.t = t + dt

        return self.plot_angle(z)
        return self.plot_particle(z)
        return self.plot_particle(z)
    
    def make_ele(self, r=100, q=1):
        p = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        mask = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        x = np.arange(-r,r+1,dtype=np.float32)
        y = np.arange(-r,r+1,dtype=np.float32)
        x, y = np.meshgrid(x, y)
        zs = x + 1j * y
        mask = np.abs(zs) <= r
        zs[mask == False] = 0
        zs = (zs / r) ** q + 0.00001
        zs = zs / np.absolute(zs)

        # torch
        zs = torch.from_numpy(zs).cuda()
        mask = torch.from_numpy(mask).cuda()
        return zs, mask
    
    def put(self, z, v, zs, mask):
        R = mask.size()[0]
        r = R // 2
        ly, lx = z.size()[2:]
        z[0, 0,ly//2-r-2*r:ly//2+r+1-2*r,lx//2-r:lx//2+r+1][mask] = zs[mask]
        v[0, 0,ly//2-r-2*r:ly//2+r+1-2*r,lx//2-r:lx//2+r+1][mask] = 0

        z[0, 0,ly//2-r+2*r:ly//2+r+1+2*r,lx//2-r:lx//2+r+1][mask] = zs[mask].conj()
        v[0, 0,ly//2-r+2*r:ly//2+r+1+2*r,lx//2-r:lx//2+r+1][mask] = 0
        return z, v

    def plot_potential(self, z):
        P = (1 - z.conj()*z)**2
        
        # rescale to -1..1
        P = 2 * P - 1
        return P.squeeze()

    def plot_particle(self, z):
        abs = z.abs() # avarage 1
        
        epsilon = 0.1
        abs = (epsilon)/(abs + epsilon)
        abs = 2 * abs - 1.0
        return abs.squeeze()

    def plot_angle(self, z):
        theta = z.angle()

        # rescale to -1..1
        theta = theta / pi
        return theta.squeeze()

    def update_numpy(self, dt=0):
        Z = self.update()
        Z = Z.to('cpu').detach().numpy().copy()

        return Z



