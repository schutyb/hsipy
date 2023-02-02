import hsitools
import tifffile
import numpy as np
import PhasorLibrary
import matplotlib.pyplot as plt

im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/01_Bead2A_405_8.lsm')

dc, g, s, _, _ = hsitools.phasor(np.asarray(im))
# PhasorLibrary.interactive(aux[0], aux[1], aux[2], 0.2, nbit=8)

x, y = hsitools.histogram_thresholding(dc, g, s, 2, 10)
