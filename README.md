# hsipy: Is a Python module to perform phasor analysis and visualization.

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


$$ G(\lambda) = \frac{\int_L I(\lambda) cos\left( 2\pi n \frac{\lambda - \lambda_i}{\lambda_{max} - \lambda_{min}} \right)}{\int_L I(\lambda)d\lambda}$$

$$ S(\lambda) = \frac{\int_L I(\lambda) sen\left( 2\pi n \frac{\lambda - \lambda_i}{\lambda_{max} - \lambda_{min}} \right)}{\int_L I(\lambda)d\lambda} j $$


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

This funtion uses matplotlib for representation. It allows user to plot the dc image and its corresponding histogram. 
In the histogram window the user is expected to pick an intensity value which will threshold the phasor plot later. 
Next it shows the phasor plot, where you can pick the circle componentes, and will create and display the pseudocolor image
and the respectives spectrums to each circle. 

![fig1](https://github.com/schutyb/rep-hsipy/blob/main/figures/int1.png)


##### Interactive2

It allows to plot the dc image and the histogram where you can pick the cut off intensity to threshold the phasor 
and later its displays the pseudoclor image created with a Hue Saturation Value color scale 
related to the phase and modulation.

![fig2](https://github.com/schutyb/rep-hsipy/blob/main/figures/int2.png)

## Authors

- [@schutyb](https://www.github.com/schutyb)


## License

[bsd-3-clause](https://choosealicense.com/licenses/bsd-3-clause/)


## Contributing

Contributions are always very well welcome. The PhasorPy library intends 
to create an open-source and collaborative community between spectroscopy 
and fluorescence microscopy users with the same functionalities as SimFCS 
but accessible and self-sustainable in the long term as other Python 
libraries and communities. 


## References

[1] Malacrida, L., Gratton, E. & Jameson, D. M. Model-free methods to study 
membrane environmental probes: A comparison of the spectral phasor and 
generalized polarization approaches. Methods Appl. Fluoresc. 3, 047001 (2015).

