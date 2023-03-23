import tifffile
import numpy as np
import hsitools
import hsi_visualization


imstack = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/paramecium/SP_paramesium_561_2laser_R1.lsm')
dc, g, s, md, ph = hsitools.phasor(imstack)  # create instance of PhasorCalculator class
hsi_stack = np.asarray(imstack)


try1 = True
if try1:
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=4, filt=True, nfilt=3, spectrums=True,
                                   hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

try2 = False
# Test interactive2, show dc image and histogram in the first window and phasor
# and pseudocolor image with the given phase and modulation range.
if try2:
    phint = np.array([40, 125])
    mdint = np.array([0.65, 0.95])
    hsi_visualization.interactive2(dc, g, s, 8, ph, phint, modulation=md, mdint=mdint,
                                   histeq=True, filt=True, nfilt=2)
