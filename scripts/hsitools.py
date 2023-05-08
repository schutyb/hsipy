import numpy as np
import colorsys


def phasor(image_stack, harmonic=1):
    """
	This function computes the average intensity image, the G and S coordinates, modulation and phase.

	:param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from numpy library.
	:param harmonic: int. The number of the harmonic where the phasor is calculated.
	harmonic range: [1, num of channels - 1]
	:return: avg: is the average intensity image
	:return: g: is mxm image with the real part of the fft.
	:return: s: is mxm imaginary with the real part of the fft.
	:return: md: is the modulus obtain with Euclidean Distance.
	:return: ph: is the phase between g and s in degrees.
	"""
    if image_stack.any():
        if isinstance(harmonic, int) and 0 < harmonic < len(image_stack):
            data = np.fft.fft(image_stack, axis=0, norm='ortho')
            dc = data[0].real
            dc = np.where(dc != 0, dc, int(np.mean(dc, dtype=np.float64)))  # change the zeros to the img average
            g = data[harmonic].real
            g /= dc
            s = data[harmonic].imag
            s /= -dc
            md = np.sqrt(g ** 2 + s ** 2)
            phaux = np.angle(data[harmonic], deg=True)
            ph = np.where(phaux < 0, phaux + 360, phaux)
            avg = np.mean(image_stack, axis=0, dtype=np.float64)
            avg = avg / avg.max() * 255
        else:
            raise ValueError(
                "harmonic indices is not integer or slice or harmonic out of range\n harmonic range: [1, ""num of channels - 1]")
        return avg, g, s, md, ph
    else:
        raise ValueError("Image stack data is an empty array")


def tilephasor(image_stack, dimx, dimy, harmonic=1):
    """
	This function compute the fft and calculate the phasor for an stack containing many tiles
	of images.
	:param dimy: images horizontal dimension
	:param dimx: images vertical dimension
	:param image_stack: image stack containing the n lambda channels
	:param harmonic: The nth harmonic of the phasor. Type int.
	:return: avg: is the average intensity image
	:return: g: is mxm image with the real part of the fft.
	:return: s: is mxm imaginary with the real part of the fft.
	:return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
	:return: ph: is the phase between g and s in degrees.
	"""

    if image_stack.any():
        if isinstance(harmonic, int) and 0 < harmonic < len(image_stack):
            dc = np.zeros([len(image_stack), dimx, dimy])
            g = np.zeros([len(image_stack), dimx, dimy])
            s = np.zeros([len(image_stack), dimx, dimy])
            md = np.zeros([len(image_stack), dimx, dimy])
            ph = np.zeros([len(image_stack), dimx, dimy])
            for i in range(len(image_stack)):
                dc[i], g[i], s[i], md[i], ph[i] = phasor(image_stack[i], harmonic=harmonic)
            return dc, g, s, md, ph
        else:
            raise ValueError(
                "harmonic indices is not integer or slice or harmonic out of range\n harmonic range: [1, ""num of channels - 1]")
    else:
        raise ValueError("Image stack data is an empty array")


def histogram_thresholding(dc, g, s, imin, imax=None):
    """
	Use this function to filter the background deleting, those pixels where the intensity value is under ic.
	:param dc: ndarray. HSI stack average intensity image.
	:param g: ndarray. G image.
	:param s: ndarray. S image.
	:param imin: Type integer. Minimum cutoff intensity value.
	:param imax: Type integer. Maximum cutoff intensity value.
	:return: x, y. Arrays contain the G and S phasor coordinates.
	"""
    if dc.any():
        if g.any():
            if s.any():
                if isinstance(imin, int):
                    aux1 = np.where(dc > imin, dc, np.zeros(dc.shape))
                    g = g.ravel()
                    s = s.ravel()
                    if imax is not None and not isinstance(imax, int):
                        raise ValueError("imax value is not an integer")
                    if imax is not None:
                        aux2 = np.where(dc < imax, np.ones(dc.shape), np.zeros(dc.shape))
                        aux = np.logical_and(aux1, aux2)
                    else:
                        aux = aux1
                    indices = np.flatnonzero(aux)
                    x = g[indices]
                    y = s[indices]
                    return x, y
                else:
                    raise ValueError("imin value is not an integer")
            else:
                raise ValueError("Empty s array")
        else:
            raise ValueError("Empty g array")
    else:
        raise ValueError("Empty dc array")


def imthreshold(im, imin, imax=None):
    """
	Threshold an image given a minimum (intensity) value, for a close interval used imax.
	:param im: image to threshold
	:param imin: left intensity value threshold
	:param imax: right intensity value threshold. It is None there is no superior cutoff intensity
	:return: image threshold
	"""
    if not im.any():
        raise ValueError("Empty image array")
    if not isinstance(imin, int):
        raise ValueError("imin value is not an integer")
    if imax is not None and not isinstance(imax, int):
        raise ValueError("imax value is not an integer")

    imt = np.where(im > imin, im, np.zeros(im.shape))
    if imax is not None:
        imt = np.where(imt < imax, imt, np.zeros(imt.shape))
    return imt


def tile_stitching(im, m, n, hper=0.05, vper=0.05, bidirectional=False):
    """
		Stitches a stack image from mxn images create an m x n only image.
	:param im: image stack to be concatenated, containing mxn images.
	:param m: number of vertical images
	:param n: number of horizontal images
	:param hper: horizontal percentage of overlap
	:param vper: vertical percentage of overlap
	:param bidirectional: Optional, set true if the image tile are bidirectional array
	:return: concatenated image
	"""
    if im.any():
        if isinstance(m, int):
            if isinstance(n, int):
                d = im.shape[1]
                aux = np.zeros([d * m, d * n])  # store the concatenated image
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
                if im.any():
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
                else:
                    raise ValueError("Empty image array")


def phase_modulation_image(ph, phinterval, md=None, mdinterval=None, outlier_cut=True, color_scale=0.95):
    """
		Given the modulation and phase it returns the pseudo color image in RGB normalizing the phase and modulation
		intro [0, 1] in order to obtain the RGB
	:param color_scale: Percentage of the phase color between 0 and 360 degrees it is used in the scale
	:param outlier_cut: Set True to set to black the phasor outliers and False to set these pixels to the max and min
	:param ph: Nd-array. Phase
	:param md: Nd-array. Modulation
	:param phinterval: array contains the max and min of phase to normalize the phase image
	:param mdinterval: array contains the max and min of modulation to normalize the modulation image
	:return: rgb the colored image in RGB space. Format numpy ndarray.
	"""
    if not ph.any():
        raise ValueError("Dimension error in phase matrix or modulation matrix")
    if md is not None:
        if not (ph.shape == md.shape):
            raise ValueError("Phase or Modulation matrix: Dimension not match")
    if not (len(phinterval) == 2):
        raise ValueError("ph interval is not 2d array")

    hsv = np.ones([ph.shape[0], ph.shape[1], 3])
    rgb = np.zeros(hsv.shape)
    if md is None:  # execute this sentence only if md is None
        for i in range(hsv.shape[0]):
            for j in range(hsv.shape[1]):
                if outlier_cut:  # cut off the outliers, set them to black if value is zero
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
    # rgb = rgb2bitmap(rgb)
    return rgb


def pseudocolor_image(dc, g, s, center, Ro, ncomp=5):
    """
		Create a matrix to see if a pixels is into the circle, using circle equation
	so the negative values of Mi means that the pixel belong to the circle and multiply
	aux1 to set zero where the avg image is under ic value
	:param ncomp: number of cursors to be used in the phasor, and the pseudocolor image.
	:param dc: ndarray. Intensity image.
	:param g:  ndarray. G image.
	:param s:  ndarray. S image.
	:param ic: intensity cut umbral. Default 0
	:param Ro: circle radius.
	:param center: ndarray containing the center coordinate of each circle.
	:return: rgba pseudocolored image.
	"""
    img = np.zeros([dc.shape[0], dc.shape[1], 3])
    ccolor = [[128, 0, 128], [0, 0, 1], [0, 1, 0], [255, 255, 0], [1, 0, 0]]  # colors are v, b, g, y, r
    for i in range(ncomp):
        M = ((g - center[i][0]) ** 2 + (s - center[i][1]) ** 2 - Ro ** 2)
        indices = np.where(M < 0)
        img[indices[0], indices[1], :3] = ccolor[i]
    return img


def avg_spectrum(hsi_stack, g, s, ncomp, Ro, center):
    """
	:param hsi_stack: hyperspectral imaging stack. Type nd numpy array
	:param g: G image containing the g coordinates of the phasor. Type nd numpy array
	:param s: S image containing the s coordinates of the phasor. Type nd numpy array
	:param ncomp: Amount of components.
	:param Ro: Radius. Type decimal.
	:param center: (g, s) Coordinates in the phasor plot to center each component. Type  numpy array
	:return: Average spectrums corresponding to each component.
	"""
    spect = np.zeros([ncomp, len(hsi_stack)])
    for i in range(ncomp):
        M = ((g - center[i][0]) ** 2 + (s - center[i][1]) ** 2 - Ro ** 2)
        indices = np.where(M < 0, np.ones(M.shape), np.zeros(M.shape))
        hsi = hsi_stack * indices
        aux = np.mean(hsi, axis=(1, 2))  # <-- vectorize the mean calculation across 2nd and 3rd axis
        aux = aux / aux.max()
        spect[i] = aux
    return spect


def rgb2bitmap(rgb):
    """
		Turn rgb image into 8 bit values.
	:param rgb: RGB image, shape (m,n,3)
	:return: image (m, n)
	"""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return (r * 6 / 256) * 36 + (g * 6 / 256) * 6 + (b * 6 / 256)


def impseudocolor(ph, md):
    """
	Creates a hsv color image with phase and modulation images.
	Set the background to black
	:param ph: (Numpy array) Image with the phase values
	:param md: (Numpy array) Image with the modulation values
	:return: Image hsv with values turn into hsv to be representable on the screen.
	''"""
    hsv = np.zeros([ph.shape[0], ph.shape[1], 3])
    hsv[..., 0] = ph / ph.max()
    hsv[..., 1] = md
    rgb = np.zeros(hsv.shape)
    rgb[ph + md == 0] = [0, 0, 0]  # <-- vectorize the black background calculation
    nonzeros = np.nonzero(ph + md)
    rgb[nonzeros] = np.array([colorsys.hsv_to_rgb(hsv[i][j][0], hsv[i][j][1], 1) for i, j in zip(*nonzeros)])
    return rgb
