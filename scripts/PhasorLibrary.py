import numpy as np
from tifffile import imwrite, memmap
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Cursor
import colorsys


# TODO add all the raise ValueError in all the function
def phasor(image_stack, harmonic=1):
    """
        This function computes the average intensity image, the G and S coordinates of the phasor.
    As well as the modulation and phase.

    :param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from
    numpy library.
    :param harmonic: int. The number of the harmonic where the phasor is calculated.
    :return: avg: is the average intensity image
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g and s in degrees.
    """

    data = np.fft.fft(image_stack, axis=0, norm='ortho')

    dc = data[0].real
    dc = np.where(dc != 0, dc, int(np.mean(dc)))  # change the zeros to the img average

    g = data[harmonic].real
    g /= dc
    s = data[harmonic].imag
    s /= -dc

    md = np.sqrt(g ** 2 + s ** 2)
    ph = np.angle(data[harmonic], deg=True) + 180
    avg = np.mean(image_stack, axis=0)

    return avg, g, s, md, ph


def phasor_tile(im_stack, dimx, dimy):
    """
        This function compute the fft and calculate the phasor for an stack containing many tiles
        of microscopy images.
    :param dimy: images horizontal dimension
    :param dimx: images vertical dimension
    :param im_stack: image stack containing the n lambda channels
    :return: avg: is the average intensity image
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g and s in degrees.
    """

    dc = np.zeros([len(im_stack), dimx, dimy])
    g = np.zeros([len(im_stack), dimx, dimy])
    s = np.zeros([len(im_stack), dimx, dimy])
    md = np.zeros([len(im_stack), dimx, dimy])
    ph = np.zeros([len(im_stack), dimx, dimy])

    for i in range(len(im_stack)):
        dc[i], g[i], s[i], md[i], ph[i] = phasor(im_stack[i], harmonic=1)

    return dc, g, s, md, ph


def generate_file(filename, gsa):
    """
    :param filename: Type string characters. The name of the file to be written, with the extension ome.tiff
    :param gsa: Type n-dimensional array holding the data to be stored.
    :return file: The file storing the data. If the filename extension was ome.tiff the file is an
    ome.tiff format.
    """
    # todo crete the compression option in order to store small data
    imwrite(filename, data=gsa)
    file = memmap(filename)
    file.flush()
    return file


def concat_d2(im, per=0.05):
    """
        Concatenate a stack of images whose dimension is 2x2xn
    :param per: percentage of overlapping
    :param im: stack image with the images to be concatenated. It is a specific 2x2 concatenation.
    :return: im_concat it is an image stack with the concatenated images.
    """

    im_concat = []

    # Caso 1 donde es una imagen de 4 partes y un solo canal
    if len(im.shape) == 3 and im.shape[0] == 4:
        d = 1024
        l = int(d * per)
        m = int(d / 2 + l)
        n = int(d / 2 - l)

        im_concat = np.zeros([d, d])

        # Concatenate 4 images excluding the interjection
        im_concat[0:n, 0:n] = im[0][0:n, 0:n]
        im_concat[0:n, m:d] = im[1][0:n, 2 * l:m]
        im_concat[m:d, 0:n] = im[2][2 * l:m, 0:n]
        im_concat[m:d, m:d] = im[3][2 * l:m, 2 * l:m]

        # Average the common zones and put them in the final concatenated image
        # vertical
        imp1 = (im[0][0:n, n:m] + im[1][0:n, 0:2 * l]) / 2
        imp2 = (im[2][2 * l:m, n:m] + im[3][2 * l:m, 0:2 * l]) / 2
        # horizontal
        imp3 = (im[0][n:m, 0:n] + im[2][0:2 * l, 0:n]) / 2
        imp4 = (im[1][n:m, 2 * l:m] + im[3][0:2 * l, 2 * l:m]) / 2
        # centre
        imp_centro = (im[0][n:m, n:m] + im[1][0:2 * l, n:m] + im[2][0:2 * l, n:m] +
                      im[3][0:2 * l, 0:2 * l]) / 4

        # Add the last part to the concatenated image
        im_concat[n:m, n:m] = imp_centro
        im_concat[0:n, n:m] = imp1
        im_concat[m:d, n:m] = imp2
        im_concat[n:m, 0:n] = imp3
        im_concat[n:m, m:d] = imp4

    # Caso 2 donde es una imagen de 4 partes y más de un canal
    elif len(im.shape) == 4 and im.shape[0] == 4:
        d = 1024
        t = im.shape[1]
        s = im.shape[2]
        l = int(d * per)
        m = int(d / 2 + l)
        n = int(d / 2 - l)

        im_concat = np.zeros([t, d, d])
        # verticales
        imp1 = np.zeros([s, n, 2 * l])
        imp2 = np.zeros([s, n, 2 * l])
        # horizontales
        imp3 = np.zeros([s, 2 * l, n])
        imp4 = np.zeros([s, 2 * l, n])
        imp_centro = np.zeros([s, 2 * l, 2 * l])

        for i in range(0, t):
            # Concatenate 4 images excluding the interjection
            im_concat[i][0:n, 0:n] = im[0][i][0:n, 0:n]
            im_concat[i][0:n, m:d] = im[1][i][0:n, 2 * l:m]
            im_concat[i][m:d, 0:n] = im[2][i][2 * l:m, 0:n]
            im_concat[i][m:d, m:d] = im[3][i][2 * l:m, 2 * l:m]

            # Average the common zones and put them in the final concatenated image
            # vertical
            imp1[i] = (im[0][i][0:n, n:m] + im[1][i][0:n, 0:2 * l]) / 2
            imp2[i] = (im[2][i][2 * l:m, n:m] + im[3][i][2 * l:m, 0:2 * l]) / 2
            # horizontal
            imp3[i] = (im[0][i][n:m, 0:n] + im[2][i][0:2 * l, 0:n]) / 2
            imp4[i] = (im[1][i][n:m, 2 * l:m] + im[3][i][0:2 * l, 2 * l:m]) / 2
            # centre
            imp_centro[i] = (im[0][i][n:m, n:m] + im[1][i][0:2 * l, n:m] + im[2][i][0:2 * l, n:m] +
                             im[3][i][0:2 * l, 0:2 * l]) / 4

            # Add the last part to the concatenated image
            im_concat[i][n:m, n:m] = imp_centro[i]
            im_concat[i][0:n, n:m] = imp1[i]
            im_concat[i][m:d, n:m] = imp2[i]
            im_concat[i][n:m, 0:n] = imp3[i]
            im_concat[i][n:m, m:d] = imp4[i]

    else:
        print('Image stack dimension is not 3 or 4. So it returns null.')

    return im_concat


def histogram_thresholding(dc, g, s, ic):
    """
        Use this function to filter the background deleting, those pixels where the intensity value is under ic.
    :param dc: ndarray. Intensity image.
    :param g:  ndarray. G image.
    :param s:  ndarray. S image.
    :param ic: intensity cut umbral.
    :return: x, y. Arrays contain the G and S phasor coordinates.
    """

    """store the coordinate to plot in the phasor"""
    aux = np.concatenate(np.where(dc > ic, dc, np.zeros(dc.shape)))
    g2 = np.concatenate(g)
    s2 = np.concatenate(s)
    x = np.delete(g2, np.where(aux == 0))
    y = np.delete(s2, np.where(aux == 0))

    return x, y


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

    return ax


def rgb_coloring(dc, g, s, ic, center, Ro):
    """
        Create a matrix to see if a pixels is into the circle, using circle equation
    so the negative values of Mi means that the pixel belong to the circle and multiply
    aux1 to set zero where the avg image is under ic value
    :param dc: ndarray. Intensity image.
    :param g:  ndarray. G image.
    :param s:  ndarray. S image.
    :param ic: intensity cut umbral.
    :param Ro: circle radius.
    :param center: ndarray containing the center coordinate of each circle.
    :return: rgba pseudocolored image.
    """
    aux1 = np.where(dc > ic, dc, np.zeros(dc.shape))
    M1 = ((g - center[0][0]) ** 2 + (s - center[0][1]) ** 2 - Ro ** 2) * aux1
    M2 = ((g - center[1][0]) ** 2 + (s - center[1][1]) ** 2 - Ro ** 2) * aux1
    M3 = ((g - center[2][0]) ** 2 + (s - center[2][1]) ** 2 - Ro ** 2) * aux1

    # img_new = np.copy(dc)
    img_new = np.zeros(dc.shape)  # todo si uso esto escrive sobre una img de fondo negro

    indices1 = np.where(M1 < 0)
    indices2 = np.where(M2 < 0)
    indices3 = np.where(M3 < 0)

    cmap = plt.cm.gray
    norm = plt.Normalize(img_new.min(), img_new.max())
    rgba = cmap(norm(img_new))
    # Set the colors
    rgba[indices1[0], indices1[1], :3] = 0, 0, 1  # blue
    rgba[indices2[0], indices2[1], :3] = 0, 1, 0  # green
    rgba[indices3[0], indices3[1], :3] = 1, 0, 0  # red

    return rgba


def phasor_plot(dc, g, s, ic=None, title=None, xlabel=None, same_phasor=False):
    """
        Plots nth phasors in the same figure.
    :param dc: image stack with all the average images related to each phasor nxn dimension.
    :param g: nxn dimension image.
    :param s: nxn dimension image.
    :param xlabel: x label for each phasor plot
    :param ic: array length numbers of g images contains the cut intensity for related to each avg img.
    :param title: (optional) the title of each phasor
    :param same_phasor: (optional) if you want to plot the same phasor with different ic set True
    :return: the phasor figure and x and y arrays containing the G and S values.
    """

    #  check that the files are correct
    global fig
    if not (len(dc) == len(g) and len(g) == len(s)):
        raise ValueError("dc, g and s dimension do not match or ic dimension is not correct")
    if not len(dc) != 0:
        raise ValueError("Some input image stack is empty")
    if ic is None:
        ic = [0]

    if len(dc) == len(ic):
        if title is None:
            title = ['Phasor']
        num_phasors = len(dc)
        # create the figures with all the phasors in each axes or create only one phasor
        if num_phasors > 1:
            fig, ax = plt.subplots(1, num_phasors, figsize=(18, 5))
            fig.suptitle('Phasor')
            for k in range(num_phasors):
                x, y = (histogram_thresholding(dc[k], g[k], s[k], ic[k]))
                phasor_circle(ax[k])
                ax[k].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
                if len(title) > 1:
                    ax[k].set_title(title[k])
                    if xlabel:
                        ax[k].set_xlabel(xlabel[k])
                if same_phasor:
                    ax[k].set_xlabel('ic' + '=' + str(ic[k]))

        elif num_phasors == 1:
            x, y = histogram_thresholding(dc, g, s, ic)
            fig, ax = plt.subplots()
            ax.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
            ax.set_title('Phasor')
            phasor_circle(ax)
        return fig
    else:
        raise ValueError("dc and ic have different length")


def circle_lines(ax, phase):
    """
        Built the figure inner and outer circle and the 45 degrees lines in the plot
    :param phase: array containing the 5 phases in degrees to plot the lines
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
    #  circle
    ax.plot(x1, list(map(yp1, x1)), color='darkgoldenrod')
    ax.plot(x1, list(map(yn1, x1)), color='darkgoldenrod')
    ax.plot(x2, list(map(yp2, x2)), color='darkgoldenrod')
    ax.plot(x2, list(map(yn2, x2)), color='darkgoldenrod')
    #  x = 0 and y = 0
    ax.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    ax.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    theta = np.pi / 180
    #  lines
    x11 = np.linspace(start=0, stop=0.34, num=100)
    x12 = np.linspace(start=0, stop=0.21, num=100)
    x13 = np.linspace(start=-0.173, stop=0, num=100)
    x14 = np.linspace(start=-0.5, stop=0, num=100)
    x15 = np.linspace(start=-0.70, stop=0, num=100)
    ax.plot(x11, np.tan(phase[0] * theta) * x11, color='mediumvioletred')
    ax.plot(x12, np.tan(phase[1] * theta) * x12, color='mediumslateblue')
    ax.plot(x13, np.tan(phase[2] * theta) * x13, color='cyan')
    ax.plot(x14, np.tan(phase[3] * theta) * x14, color='lime')
    ax.plot(x15, np.tan(phase[4] * theta) * x15, color='red')
    return ax


def phasor_figure(x, y, phases=None, circle_plot=False, phases_lines=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle('Phasor')
    ax.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    if circle_plot:
        phasor_circle(ax)
    if phases_lines:
        circle_lines(ax, phases)
    return fig


def interactive(dc, g, s, Ro, nbit):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot the phasor.
    To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param nbit: bits oof the image
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param Ro: radius of the circle to select pixels in the phasor
    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """
    nbit = 2**nbit
    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

    ax[0, 0].imshow(dc, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Average intensity image')
    ax[0, 1].hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax[0, 1].set_yscale("log")
    cursor = Cursor(ax[0, 1], horizOn=False, vertOn=True, color='darkgoldenrod')
    ax[0, 1].set_title('Average intensity image histogram')

    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])

    x, y = histogram_thresholding(dc, g, s, ic)  # x y contain g and s coordinate to pass to hist2d function

    phasor_circle(ax[1, 0])
    ax[1, 0].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    ax[1, 0].set_title('Phasor')

    center = plt.ginput(3, timeout=0)  # store the center of each circle
    rgba = rgb_coloring(dc, g, s, ic, center, Ro)
    ax[1, 1].imshow(rgba)
    ax[1, 1].set_title('Pseudocolor image')
    ax[1, 1].axis('off')
    plt.show()

    return fig


def histogram_line(Ro, g, s, dc, ic, N=100, print_fractions=False):
    """
        This function plot the histogram between two components in the phasor.
    :param Ro: int. radius of the circle to select pixels in the phasor
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param dc: average intensity image. ndarray
    :param ic: intensity background cutoff
    :param N: number of division in the line between components.
    :param print_fractions: (optional) set true if you want to print the % component of each pixel on terminal.
    :return: fig. Figure with phasor and the pixel histogram.
    """
    x_c, y_c = histogram_thresholding(dc, g, s, ic)
    fig, ax = plt.subplots(1, 2)
    ax[0].hist2d(x_c, y_c, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    ax[0].set_title('Phasor - components determination')
    ax[0].set_xlabel('G')
    ax[0].set_ylabel('S')
    phasor_circle(ax[0])

    p = plt.ginput(2, timeout=False)

    ax[0].annotate('Componente A', xy=(p[0][0], p[0][1]),
                   xytext=(p[0][0] + 0.25 * abs(p[0][0]), p[0][1] + 0.25 * abs(p[0][1])),
                   arrowprops=dict(facecolor='black', arrowstyle='simple'))

    ax[0].annotate('Componente B', xy=(p[1][0], p[1][1]),
                   xytext=(p[1][0] + 0.25 * abs(p[1][0]), p[1][1] + 0.25 * abs(p[1][1])),
                   arrowprops=dict(facecolor='black', arrowstyle='simple'))

    ax[0].plot((p[0][0], p[1][0]), (p[0][1], p[1][1]), 'k')

    circle1 = plt.Circle(p[0], radius=Ro, color='k', fill=False)
    plt.gca().add_artist(circle1)
    circle2 = plt.Circle(p[1], radius=Ro, color='k', fill=False)
    plt.gca().add_artist(circle2)

    circle1 = plt.Circle(p[0], radius=Ro / 10, color='k')
    plt.gca().add_artist(circle1)
    circle2 = plt.Circle(p[1], radius=Ro / 10, color='k')
    plt.gca().add_artist(circle2)

    "x and y are the G and S coordinates"
    x = np.linspace(min(p[0][0], p[1][0]), max(p[0][0], p[1][0]), N)
    y = p[0][1] + ((p[0][1] - p[1][1]) / (p[0][0] - p[1][0])) * (x - p[0][0])

    a = np.array([[p[0][0], p[1][0]], [p[0][1], p[1][1]]])
    mf = np.zeros([len(x), 2])

    for i in range(0, len(x)):
        gs = np.array([x[i], y[i]])
        f = np.linalg.solve(a, gs)
        mf[i][0] = round(f[0], 2)
        mf[i][1] = round(f[1], 2)

    fx = np.linspace(0, 1, N) * 100

    """
    calculate the amount of pixels related to a point in the segment
    calculate the distance between x and x_c the minimal distance means
    that we have found the G coordinate, the same for S
    """
    hist_p = np.zeros(N)
    aux1 = np.where(dc > ic, dc, np.zeros(dc.shape))

    for ni in range(0, N):
        """
        create a matrix to see if a pixels is into the circle, using circle equation
        so the negative values of Mi means that the pixel belong to the circle
        """

        m1 = ((g - x[ni]) ** 2 + (s - y[ni]) ** 2 - Ro ** 2) * aux1
        indices = np.where(m1 < 0)
        hist_p[ni] = len(indices[0])

    ax[1].plot(fx, hist_p)
    ax[1].set_title('pixel histogram')
    ax[1].grid()

    plt.show()

    if print_fractions:
        print('Componente A  \t Componente B')
        for i in range(0, len(x)):
            print(mf[i][0], '\t\t', mf[i][1])

    return ax


def concatenate(im, m, n, hper=0.07, vper=0.05, bidirectional=False):
    """
        This function concatenate a stack image from mxn images create an m x n only image.
    :param im: image stack to be concatenated, containing mxn images. The dimension is 1 x (nxm).
    :param m: number of vertical images
    :param n: number of horizontal images
    :param hper: horizontal percentage of overlap
    :param vper: vertical percentage of overlap
    :param bidirectional: Optional, set true if the image tile are bidirectional array
    :return: concatenated image
    """
    d = im.shape[1]
    aux = np.zeros([d * m, d * n])  # create the matrix to store the concatenated image
    # Horizontal concatenate
    i = 0
    j = 0
    while j < m * n:
        if bidirectional and ((j / n) % 2 == 1):
            aux[i * d: i * d + d, 0:d] = im[j + (n - 1)][0:, 0:d]  # store the first image horizontally
        else:
            aux[i * d: i * d + d, 0:d] = im[j][0:, 0:d]  # store the first image horizontally
        k = 1
        acum = 0
        if bidirectional and ((j / n) % 2 == 1):
            while k < n:
                ind1 = round(((1 - vper) + acum) * d)
                ind2 = round(ind1 + vper * d)
                ind3 = round(ind2 + (1 - vper) * d)
                aux[i * d:i * d + d, ind1:ind2] = (aux[i * d:i * d + d, ind1:ind2] + im[j + (n - k - 1)][0:,
                                                                                     0:round(vper * d)]) / 2
                aux[i * d:i * d + d, ind2:ind3] = im[j + (n - k - 1)][0:, round(vper * d):d]
                acum = (1 - vper) + acum
                k = k + 1
        else:
            while k < n:
                ind1 = round(((1 - vper) + acum) * d)
                ind2 = round(ind1 + vper * d)
                ind3 = round(ind2 + (1 - vper) * d)
                aux[i * d:i * d + d, ind1:ind2] = (aux[i * d:i * d + d, ind1:ind2] + im[j + k][0:,
                                                                                     0:round(vper * d)]) / 2
                aux[i * d:i * d + d, ind2:ind3] = im[j + k][0:, round(vper * d):d]
                acum = (1 - vper) + acum
                k = k + 1
        i = i + 1
        j = j + n

    # Vertical concatenate
    img = np.zeros([round(d * (m - hper * (m - 1))), round(d * (n - hper * (n - 1)))])
    img[0:d, 0:] = aux[0:d, 0:img.shape[1]]
    k = 1
    while k < m:
        #  indices de la matrix aux para promediar las intersecciones
        ind1 = round(d * (k - hper))
        ind2 = round(d * k)
        ind3 = round(d * (k + hper))
        ind4 = round(d * (k + 1))
        #  indices de la nueva matriz donde se almacena la imagen final
        i1 = round(k * d * (1 - hper))
        i2 = round(i1 + d * hper)
        i3 = round(i2 + d * (1 - hper))

        img[i1:i2, 0:] = (aux[ind1:ind2, 0:img.shape[1]] + aux[ind2:ind3, 0:img.shape[1]]) / 2
        img[i2:i3, 0:] = aux[ind3:ind4, 0:img.shape[1]]
        k = k + 1

    return img


def psnr(img_optimal, img):
    """
    :param img_optimal: Nd-array image. Should contain the gold standard image.
    :param img: Nd-array image. Image to be compared
    :return: Float value. Peak Signal to Noise Ratio.
    """
    if img_optimal.shape and img.shape and (img_optimal.shape == img.shape):
        mse = np.mean(abs(img_optimal - img) ** 2)
        psnr_aux = 10 * np.log10((255 ** 2) / mse)
    else:
        raise ValueError("Images dimension do not much")
    return psnr_aux


def segment_thresholding(hist, bins, per, complete_hist=False):
    """
        Given a histogram and a percentage the function returns a cut histogram
        preserving the values where the histogram is over the percentage * max(hist)
    :param hist: One dimensional array, length n
    :param bins: One dimensional array, length n
    :param per: percentage of the maximum hist value to be considered
    :return: acum: the array containing the threshold bins
    """
    if not len(hist):
        raise ValueError("histogram dimension or type is not correct (dim = 1xn and type: ndarray)")
    if not len(bins):
        raise ValueError("bins dimension or type is not correct (dim = 1xn and type: ndarray)")
    if not (len(hist) == len(bins)):
        raise ValueError("histogram array length and bins array length do not much")
    if not (0 <= per <= 1):
        raise ValueError("per must be a float in [0;1]")
    else:
        acum = []
        for ind in range(len(hist)):
            if hist[ind] > round(per * max(hist)):
                acum.append(bins[ind])
    if complete_hist:
        return np.asarray(acum), np.asarray([min(acum), max(acum)])
    else:
        return np.asarray([min(acum), max(acum)])


def im_thresholding(im, x1, x2):
    """
        Considering an image whose entry values are [0, 255]. Threshold an image from x1 to x2
    :param im: Nd-array contains the original image to be threshold
    :param x1: float value, minimal value of the left side
    :param x2: float value, maximal value of the right side
    :return: Nd-array image threshold
    """
    if not im.shape:
        raise ValueError("Input image is not dimensionally correct")
    # elif not (x1.isdigit() and x2.isdigit()): todo check if x1 and x2 are digits
    # raise ValueError("x1 or x2 are not float type")
    else:
        aux = np.where(im != 0, im, 100 * abs(np.max(im)))
        aux = np.where(aux < x1, 0, aux)
        aux = np.where(aux == 100 * abs(np.max(im)), 0, aux)
        aux = np.where(aux > x2, 0, aux)
    return aux


def colored_image(ph, phinterval, md=None, mdinterval=None, outlier_cut=True, color_scale=0.92):
    """
        Given the modulation and phase it returns the pseudo color image in RGB normalizing the phase and modulation
        intro [0, 1] in order to obtain the RGB
    :param color_scale: Percentage of the phase color between 0 and 360 degrees it is used in the scale
    :param outlier_cut: Set True to set to black the phasor outliers and False to set these pixels to the max and min
    :param ph: Nd-array. Phase
    :param md: Nd-array. Modulation
    :param phinterval: array contains the max and min of phase to normalize the phase image
    :param mdinterval: array contains the max and min of modulation to normalize the modulation image
    :return: rgb the colored image in RGB space
    """
    if not (len(ph.shape) == 2):
        raise ValueError("Dimension error in phase matrix or modulation matrix")
    if md:
        if not (ph.shape == md.shape):
            raise ValueError("Phase or Modulation matrix: Dimension not match")
    if not (len(phinterval) == 2):
        raise ValueError("ph interval is not 2d array")

    hsv = np.ones([ph.shape[0], ph.shape[1], 3])
    rgb = np.zeros(hsv.shape)
    if md is None:  # execute this sentence only if md is None
        for i in range(hsv.shape[0]):
            for j in range(hsv.shape[1]):
                if outlier_cut:  # cut off the outliers so set them to black value is zero
                    if phinterval[0] <= ph[i][j] <= phinterval[1]:
                        hsv[i][j][0] = color_scale * (ph[i][j] - phinterval[0]) / abs(phinterval[0] - phinterval[1])
                    else:
                        hsv[i][j][:] = 0, 0, 0
                else:  # in this case the outliers are put into the extremes 0 phase and maximum phase
                    if phinterval[0] <= ph[i][j] <= phinterval[1]:
                        hsv[i][j][0] = color_scale * (ph[i][j] - phinterval[0]) / abs(phinterval[0] - phinterval[1])
                    elif ph[i][j] == phinterval[0]:
                        hsv[i][j][0] = 0
                    elif ph[i][j] == phinterval[1]:
                        hsv[i][j][0] = color_scale
                rgb[i][j][:] = colorsys.hsv_to_rgb(hsv[i][j][0], hsv[i][j][1], hsv[i][j][2])
    else:
        for i in range(hsv.shape[0]):
            for j in range(hsv.shape[1]):
                if outlier_cut:
                    if (phinterval[0] <= ph[i][j] <= phinterval[1]) and (mdinterval[0] <= md[i][j] <= mdinterval[1]):
                        hsv[i][j][0] = color_scale * (ph[i][j] - phinterval[0]) / abs(phinterval[0] - phinterval[1])
                        hsv[i][j][1] = (md[i][j] - mdinterval[0]) / abs(mdinterval[0] - mdinterval[1])
                    else:
                        hsv[i][j][:] = (0, 0, 0)
                else:
                    if phinterval[0] <= ph[i][j] <= phinterval[1]:
                        hsv[i][j][0] = color_scale * (ph[i][j] - phinterval[0]) / abs(phinterval[0] - phinterval[1])
                    elif ph[i][j] == phinterval[0]:
                        hsv[i][j][0] = 0
                    elif ph[i][j] == phinterval[1]:
                        hsv[i][j][0] = color_scale

                    if mdinterval[0] <= md[i][j] <= mdinterval[1]:
                        hsv[i][j][1] = (md[i][j] - mdinterval[0]) / abs(mdinterval[0] - mdinterval[1])
                    elif md[i][j] == mdinterval[0]:
                        hsv[i][j][1] = 0
                    elif md[i][j] == mdinterval[1]:
                        hsv[i][j][1] = 1
                rgb[i][j][:] = colorsys.hsv_to_rgb(hsv[i][j][0], hsv[i][j][1], hsv[i][j][2])
    return rgb


def phasor_threshold(g, s, md, ph, phinterval, mdinterval):
    """
        Return the phasor coordinates G and S after thresholding.
    :param mdinterval: 1d array modulation values interval
    :param phinterval: 1d array phase interval
    :param md: nd-array modulation image
    :param ph: nd-array phase image
    :param g: nxn dimension image.
    :param s: nxn dimension image.
    :return: the phasor figure and x and y arrays containing the G and S values.
    """

    #  check that the files are correct
    if not len(g) == len(s):
        raise ValueError("g and s dimensions do not match")

    if not len(g) != 0:
        raise ValueError("Some input image stack is empty")

    else:
        ph2 = im_thresholding(ph, phinterval[0], phinterval[1])
        md2 = im_thresholding(md, mdinterval[0], mdinterval[1])
        lut = np.where((ph2 * md2) == 0, 0, 1)
        aux = g * s * lut
        g2 = np.concatenate(g * lut)
        s2 = np.concatenate(s * lut)
        x = np.delete(g2, np.where(aux != 0))
        y = np.delete(s2, np.where(aux != 0))

    return x, y


def phase_correlation(a, b):
    g_a = np.fft.fft2(a)
    g_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(g_b)
    r = g_a * conj_b
    r /= np.absolute(r)
    return np.fft.ifft2(r).real
