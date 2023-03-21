import tifffile
import numpy as np
import hsitools
import hsi_visualization

im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/paramecium/SP_paramesium_561_2laser_R2.lsm')
dc, g, s, modulation, phase = hsitools.phasor(np.asarray(im))

try1 = False
if try1:
    hsi_visualization.interactive1(dc, g, s, 0.075, nbit=8, ncomp=5, filt=True, nfilt=3)

hsi_stack = np.asarray(im)

try2 = False
if try2:
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=4, filt=True, nfilt=3, spectrums=True,
                                   hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

try3 = True
if try3:
    phint = np.array([45, 135])
    mdint = np.array([0.5, 0.95])
    hsi_visualization.interactive2(dc, g, s, 8, -phase, modulation, phint, mdint,
                                   histeq=True, filt=True, nfilt=2)

a = False
if a:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=45, vmax=180)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    # cb1.set_label('Degrees')
    fig.show()
    plt.show()

