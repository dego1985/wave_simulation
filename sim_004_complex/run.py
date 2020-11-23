#!/usr/bin/env python
import torch
import numpy as np

import module.plot as pl
import module.sim as sim

print('cudnn  :', torch.backends.cudnn.version())

W = 1000
N = (W, W)
wave = sim.wave(N, dx=5/W)
mesh = pl.mesh(wave)
pl.plot3d(mesh)