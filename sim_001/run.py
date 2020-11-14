#!/usr/bin/env python

import module.plot as pl
import module.sim as sim

wave = sim.wave()
mesh = pl.mesh(wave)
pl.plot3d(mesh)
