# hsipy: Is a Python module to perform phasor analysis and and visualize it.

Hyperspectral imaging (HSI) have become paramount 
in biomedical science. The power of the combination between traditional 
imaging and spectroscopy opens the possibility to address information 
inaccessible before. For bioimaging analysis of these data, the Phasor 
Plots are tools that help the field because of their straightforward 
approach. Thus it is becoming a key player in democratizing access to HSI, 
and improve open source software for bioimaging communities.


hsipy is a module for HSI data analysis using the phasor approach. 
The phasor approach was developed as model free method 
and relies on the Fourier Transform properties.


## Documentation


### Phasor Analysis 
Considering an hyperspectral image stack, the fluorescence spectra at each pixel can be
transformed in phasor coordinates (G (λ)) and (S (λ)) as described in the following 
equations. I(λ) represent the intensity at every wavelength (channel), n is the 
number of the harmonic and λ i the initial wavelength. The, x and y coordinates 
are plotted in the spectral phasor plot.

![eq1](https://github.com/bschuty/PhasorPy/blob/main/Figures/equation_spectral.png)


The angular position in the spectral phasor plot relates to the center of mass of 
the emission spectrum and the modulus depends on the spectrum’s full width at 
the half maximum (FWHM). For instance, if the spectrum is broad its location 
should be close to the center. Otherwise, if there is a red shift in the spectrum,
its location will move counterclockwise toward increasing angle from position
(1, 0). Spectral phasors have the same vector properties as lifetime phasors. 
A detailed description of the spectral phasor plot properties can be found in 
Malacrida et al. 1. 


## Installation

```bash
  pip install hsipy
  conda install hsipy
```
    
## Demo

### Phasor Plot

##### Interactive1

This funtion relays on matplotlib. It allows user to plot the dc image and its corresponding histogram. 
In the histogram window the user is expected to pick an intensity value which will threshold the phasor plot later. 
Next it shows the phasor plot, where you can pick the circle componentes, and will create and display the pseudocolor image
and the respectives spectrums to each circle. 

![fig1](https://github.com/schutyb/rep-hsipy/blob/main/figures/int1.png)
