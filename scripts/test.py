import tifffile
import numpy as np
import hsitools
import hsi_visualization


imstack = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/paramecium/SP_paramecium.lsm')
dc, g, s, md, ph = hsitools.phasor(imstack)  # create instance of PhasorCalculator class
hsi_stack = np.asarray(imstack)

try1 = False
if try1:
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=4, filt=True, nfilt=3, spectrums=True,
                                   hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

try2 = True
if try2:
    phint = np.array([225, 315])
    mdint = np.array([0.7, 1])
    hsi_visualization.interactive2(dc, g, s, 8, ph, phint, modulation=md, mdint=mdint, histeq=False, filt=True, nfilt=2)
