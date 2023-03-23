import tifffile
import numpy as np
import hsitools
import hsi_visualization
import matplotlib.pyplot as plt

imstack = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/melpig_10.lsm')
dc, g, s, md, ph = hsitools.phasor(imstack)  # create instance of PhasorCalculator class

try1 = False
if try1:
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=3, filt=True, nfilt=3)

hsi_stack = np.asarray(imstack)

try2 = False
if try2:
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=4, filt=True, nfilt=3, spectrums=True,
                                   hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

try3 = True
if try3:
    phint = np.array([90, 180])
    mdint = np.array([0.4, 0.7])
    hsi_visualization.interactive2(dc, g, s, 8, ph, phint, modulation=md, mdint=mdint,
                                   histeq=True, filt=True, nfilt=2)

a = False
if a:
    phint = np.array([90, 135])
    mdint = np.array([0, 0.7])

    rgb = hsitools.phase_modulation_image(ph, phint, md, mdint)

    fig, ax = plt.subplots()
    im = ax.imshow(rgb)
