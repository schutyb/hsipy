import tifffile
import numpy as np
import hsitools
import hsi_visualization
import matplotlib.pyplot as plt

im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/melpig_10.lsm')
dc, g, s, modulation, phase = hsitools.phasor(np.asarray(im))

inte1 = False
# test the function without plotting the spectrums related to each circle
if inte1:
    hsi_visualization.interactive1(dc, g, s, 0.075, nbit=8, ncomp=5, filt=True, nfilt=2)

hsi_stack = np.asarray(im)

inte2 = False
if inte2:
    hsi_visualization.interactive1(dc, g, s, 0.075, nbit=8, ncomp=3, filt=True, nfilt=2, spectrums=True,
                                   hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

inte3 = True
if inte3:
    phint = np.array([100, 160])
    mdint = np.array([0.4, 0.6])
    hsi_visualization.interactive2(dc, g, s, 8, phase, modulation, phint, mdint,
                                   histeq=True, filt=True, nfilt=2)


