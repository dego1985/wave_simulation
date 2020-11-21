#!/usr/bin/env python
import torch
import numpy as np

import module.plot as pl
import module.sim as sim

print('cudnn  :', torch.backends.cudnn.version())

W = 300
H = 40
N = (W, W, H)
wave = sim.wave(N, dx=5/W, z_boundary=-1)
mesh = pl.mesh(wave)
pl.plot3d(mesh)