import tools
import tifffile
import numpy as np
import PhasorLibrary
import matplotlib.pyplot as plt

im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsi/data/01_Bead2A_405_16.lsm')

aux = tools.phasor(im)
# PhasorLibrary.interactive(aux[0], aux[1], aux[2], 0.1, nbit=8)

