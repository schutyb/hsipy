import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Cursor
import hsitools


# The following two functions define the phasor figure: circle and cluster
def phasor_circle(ax):
    """
        Built the figure inner and outer circle and the 45 degrees lines in the plot
    :param ax: axis where to plot the phasor circle.
    :return: the axis with the added circle.
    """
    x1 = np.linspace(start=-1, stop=1, num=500)
    yp1 = lambda x1: np.sqrt(1 - x1 ** 2)
    yn1 = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    yp2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    yn2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)
    ax.plot(x1, list(map(yp1, x1)), color='darkgoldenrod')
    ax.plot(x1, list(map(yn1, x1)), color='darkgoldenrod')
    ax.plot(x2, list(map(yp2, x2)), color='darkgoldenrod')
    ax.plot(x2, list(map(yn2, x2)), color='darkgoldenrod')
    ax.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    ax.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    ax.scatter(x4, x4, marker='_', color='darkgoldenrod')
    ax.scatter(x4, -x4, marker='_', color='darkgoldenrod')
    ax.annotate('0º', (1, 0), color='darkgoldenrod')
    ax.annotate('180º', (-1, 0), color='darkgoldenrod')
    ax.annotate('90º', (0, 1), color='darkgoldenrod')
    ax.annotate('270º', (0, -1), color='darkgoldenrod')
    ax.annotate('0.5', (0.42, 0.28), color='darkgoldenrod')
    ax.annotate('1', (0.8, 0.65), color='darkgoldenrod')
    return ax


def phasor_figure(x, y, circle_plot=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle('Phasor')
    ax.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    if circle_plot:
        phasor_circle(ax)
    return fig


# The following 2 functions are a set of interactive functions to plot and performe phasor analysis
def interactive1(dc, g, s, Ro, nbit, histeq=True, ncomp=5, filt=False, nfilt=0, spectrums=False,
                 hsi_stack=None, lamd=None):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot the phasor.
    To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param nbit: bits of the image.
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param Ro: radius of the circle to select pixels in the phasor

    :param lamd: Lamba array containing the wavelenght. numpy array. Optional
    :param hsi_stack: HSI stack to plot the spectrums of each circle regions.
    :param spectrums: set True to plot the average spectrum of each circle. Optional
    :param nfilt: amount of times to filt G and S images. Optional
    :param filt: Apply median filter to G and S images, before the dc threshold. Optional
    :param ncomp: number of cursors to be used in the phasor, and the pseudocolor image. Default 5.
    :param histeq: equalize histogram used in dc image for a better representation.
            Its only applies for dc when plotting it. Optional

    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """
    if histeq:
        from skimage.exposure import equalize_adapthist
        auxdc = equalize_adapthist(dc / dc.max())
    else:
        auxdc = dc
    if filt:
        from skimage.filters import median
        for i in range(nfilt):
            g = median(g)
            s = median(s)
    nbit = 2 ** nbit

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(auxdc, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Average intensity image')
    ax2.hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax2.set_yscale("log")
    ax2.set_title('Average intensity image histogram')
    cursor = Cursor(ax2, horizOn=False, vertOn=True, color='darkgoldenrod')
    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])
    x, y = hsitools.histogram_thresholding(dc, g, s, ic)

    figp, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    phasor_circle(ax3)
    phasorbar = ax3.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
                           range=[[-1, 1], [-1, 1]])
    ax3.set_title('Phasor', pad=20)
    plt.sca(ax3)
    plt.xticks([-1, 0, 1], ['-1', '0', '1'])
    plt.yticks([-1, 0, 1], ['-1', '0', '1'])
    fig.colorbar(phasorbar[3], ax=ax3)
    center = plt.ginput(ncomp, timeout=0)  # get the circle centers
    ccolor = ['darkviolet', 'blue', 'green', 'yellow', 'red']
    for i in range(ncomp):
        circle = plt.Circle((center[i][0], center[i][1]), Ro, color=ccolor[i], fill=False)
        ax3.add_patch(circle)

    g = np.where(dc > ic, g, dc * np.nan)
    s = np.where(dc > ic, s, dc * np.nan)
    rgba = hsitools.pseudocolor_image(dc, g, s, center, Ro, ncomp=ncomp)
    fig2, ax4 = plt.subplots(1, 1, figsize=(8, 8))
    ax4.imshow(rgba)
    ax4.set_title('Pseudocolor image')
    ax4.axis('off')

    if spectrums:
        spect = hsitools.avg_spectrum(hsi_stack, g, s, ncomp, Ro, center)
        plt.figure(figsize=(12, 6))
        for i in range(ncomp):
            if lamd.any():
                plt.plot(lamd, spect[i], ccolor[i])
            else:
                plt.plot(spect[i], ccolor[i])
        plt.grid()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Normalize intensity')
        plt.title('Average Components Spectrums')
    plt.show()
    return fig


def interactive2(dc, g, s, nbit, phase, phint, modulation, mdint, histeq=True, filt=False, nfilt=0):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot
    the phasor. To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param phint:
    :param mdint:
    :param modulation:
    :param phase:
    :param nfilt: amount of times to filt G and S images.
    :param filt: Apply median filter to G and S images, before the dc threshold.
    :param histeq: equalize histogram used in dc image for a better representation.
    Its only applies for dc when plotting it.
    :param nbit: bits oof the image
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """
    if histeq:
        from skimage.exposure import equalize_adapthist
        auxdc = equalize_adapthist(dc / dc.max())
    else:
        auxdc = dc
    if filt:
        from skimage.filters import median
        for i in range(nfilt):
            g = median(g)
            s = median(s)
            phase = median(phase)
            modulation = median(modulation)
    nbit = 2 ** nbit

    # First figure plots dc image and its histogram
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(auxdc, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Average intensity image')
    ax2.hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax2.set_yscale("log")
    ax2.set_title('Average intensity image histogram')
    cursor = Cursor(ax2, horizOn=False, vertOn=True, color='darkgoldenrod')
    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])
    x, y = hsitools.histogram_thresholding(dc, g, s, ic)
    phase = np.where(dc > ic, phase, np.zeros(phase.shape))
    if modulation.any():
        modulation = np.where(dc > ic, modulation, np.zeros(modulation.shape))

    # Phasor plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    phasor_circle(ax3)
    ax3.set_title('Phasor Plot', pad=20)
    plt.sca(ax3)
    plt.xticks([-1, 0, 1], ['-1', '0', '1'])
    plt.yticks([-1, 0, 1], ['-1', '0', '1'])
    aux = plt.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    fig2.colorbar(aux[3], ax=ax3)
    ax3.axis('off')

    # Pseudocolor ph-md image and colorbar
    pseudocolor = hsitools.phase_modulation_image(phase, phint, md=modulation, mdinterval=mdint)
    auxphase = np.asarray(np.meshgrid(np.arange(phint[0], phint[1]), np.arange(phint[0], phint[1])))[0]
    auxmd = np.asarray(np.meshgrid(np.linspace(mdint[0], mdint[1], abs(phint[1]-phint[0])),
                                   np.linspace(mdint[0], mdint[1], abs(phint[1]-phint[0]))))[0].transpose()
    pseudo_colorbar = hsitools.phase_modulation_image(auxphase, phint, md=auxmd, mdinterval=mdint)

    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6))
    ax4.imshow(pseudocolor)
    ax4.set_title('Pseudocolor Image')
    ax4.axis('off')
    ax5.imshow(pseudo_colorbar)
    ax5.grid()
    plt.sca(ax5)
    plt.xticks(np.round(np.linspace(0, abs(phint[0]-phint[1]), 10)),
               list(np.round(np.linspace(0, abs(phint[0]-phint[1]), 10))) + phint[0])
    plt.yticks(np.round(np.linspace(0, abs(phint[0]-phint[1]), 10)),
               list(np.round(np.linspace(0, abs(mdint[0]-mdint[1]), 10), 2)) + mdint[0])
    ax5.set_title('HSV Scale for pseudocolor image')
    ax5.set_xlabel('Phase [Degrees]')
    ax5.set_ylabel('Modulation')
    plt.show()
    return fig1, fig2, fig3
