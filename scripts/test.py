import tifffile
import numpy as np
import hsitools
import hsi_visualization


imstack = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/paramecium/SP_paramecium.lsm')
# imstack = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/melpig_10.lsm')
dc, g, s, md, ph = hsitools.phasor(imstack)
hsi_stack = np.asarray(imstack)

try1 = False
if try1:
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=4, filt=True, nfilt=3, spectrums=True,
                                   hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

try2 = False
if try2:
    phint = np.array([225, 315])
    mdint = np.array([0.3, 0.8])
    hsi_visualization.interactive2(dc, g, s, 8, ph, phint, modulation=md, mdint=mdint, histeq=False, filt=True, nfilt=2)


tileim = False
if tileim:
    imstack = imstack = tifffile.imread('/home/bruno/Documentos/Proyectos/hsipy/data/SP_corn_stem_tile_3x3_bidir.lsm')
    phasor = hsitools.tilephasor(imstack, 512, 512)

    dc = hsitools.tile_stitching(phasor[0], 3, 3, bidirectional=True)
    g = hsitools.tile_stitching(phasor[1], 3, 3, bidirectional=True)
    s = hsitools.tile_stitching(phasor[2], 3, 3, bidirectional=True)
    md = hsitools.tile_stitching(phasor[3], 3, 3, bidirectional=True)
    ph = hsitools.tile_stitching(phasor[4], 3, 3, bidirectional=True)

    try3 = True
    if try3:
        hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=5, filt=True, nfilt=2, spectrums=False,
                                       hsi_stack=hsi_stack, lamd=np.arange(418, 718, 10))

    try4 = True
    if try4:
        phint = np.array([225, 315])
        mdint = np.array([0.7, 1])
        hsi_visualization.interactive2(dc, g, s, 8, ph, phint, modulation=md, mdint=mdint, histeq=False, filt=True,
                                       nfilt=2)
