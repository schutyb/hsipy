import tifffile
import numpy as np
import hsitools
import hsi_visualization
import matplotlib.pyplot as plt

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

try3 = False
if try3:
    phint = np.array([45, 135])
    mdint = np.array([0.6, 0.95])
    hsi_visualization.interactive2(dc, g, s, 8, -phase, phint, modulation=modulation, mdint=mdint,
                                   histeq=True, filt=True, nfilt=2)

a = False
if a:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    phint = np.array([45, 135])
    mdint = np.array([0.5, 0.95])

    rgb = hsitools.phase_modulation_image(-phase, modulation, phint, mdint)

    fig, ax = plt.subplots()
    im = ax.imshow(rgb, cmap='')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 3, 1]})
    # gridspec_kw={'width_ratios': [3, 1]}
    phasor_circle(ax3)
    ax3.set_title('Phasor')
    ax3.contour(counts.transpose(), extent=[xb.min(), xb.max(), yb.min(), yb.max()],
                linewidths=2, cmap='gray')
    plt.sca(ax3)
    plt.xticks([-1, 0, 1], ['-1', '0', '1'])
    plt.yticks([-1, 0, 1], ['-1', '0', '1'])

    # fig2.tight_layout()

    auxph = np.asarray(np.meshgrid(np.arange(0, 360), np.arange(0, 360))[0]).transpose()
    auxmd = np.asarray(np.meshgrid(np.linspace(0, 1, 360), np.linspace(0, 1, 360))[0])
    colorbar = hsitools.phase_modulation_image(auxph, np.asarray([0, 360]), md=auxmd,
                                               mdinterval=np.asarray([0, 1]))
    pseudocolor = hsitools.phase_modulation_image(phase, phint, md=modulation, mdinterval=mdint)

    ax4.imshow(pseudocolor)
    ax5.imshow(colorbar)
    ax4.axis('off')
    plt.sca(ax5)
    plt.xticks([0, 360], [str(phint[0]), str(phint[1])])
    plt.yticks([0, 360], [str(mdint[1]), str(mdint[0])])
    ax5.set_xlabel('Phase [Degrees]')
    ax5.set_ylabel('Modulation')
    plt.show()

fig3 = plt.figure(figsize=(12, 5), constrained_layout=True)
widths = [5, 5, 1]
heights = [5, 5, 1]
gs = fig3.add_gridspec(3, 3, width_ratios=widths,
                       height_ratios=heights)
f3_ax1 = fig3.add_subplot(gs[:, 0])
f3_ax1.set_title('1')
f3_ax1 = fig3.add_subplot(gs[:, 1])
f3_ax1.set_title('2')
f3_ax1 = fig3.add_subplot(gs[:, 2])
f3_ax1.set_title('3')
plt.show()
