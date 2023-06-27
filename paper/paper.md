---
title: 'hsipy: A Python module to perform and visualize phasor analysis in Hyperspectral Imaging'
tags:
  - Python
  - Phasors
  - Hyperspectral Imaging
  - Bioimaging
authors:
  - name: Bruno Schuty
    orcid: 0000-0002-2293-933X
    equal-contrib: False
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Leonel Malacrida
    equal-contrib: False # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2, 3"
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Institut Pasteur de Montevideo, Uruguay.
   index: 1
 - name: Unidad de Bioimagenología Avanzada, Uruguay.
   index: 2
 - name: Hospital de Clínicas, Montevideo, Uruguay.
   index: 3
date: 1 July 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Hyperspectral imaging (HSI) have become paramount in biomedical science. 
The power of the combination between traditional imaging and spectroscopy 
opens the possibility to address information inaccessible before. 
For bioimaging analysis of these data, the Phasor Plots are tools that 
help the field because of their straightforward approach. 
Thus it is becoming a key player in democratizing access to HSI, 
and improve open source software for bioimaging communities.

hsipy is a module for HSI data analysis using the phasor approach. 
The phasor approach was developed as model free method and relies 
on the Fourier Transform properties.

# Statement of need

`hsipy` is a Python module for phasor analysis in Hyperspectral imaging. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. Firthermothe it allows us to 
integrate many libraries thta performe numercial analysis, matrix computation and 
image processing, making its very useful for specify domains like hyperspectral imaging. 
This library was designed to provide a simple but powerful way to performe phasor analysis 
with the Python language, mainly apply to biophysics domains. It has huge applications in 
fluorescence microscopy and spectroscopy `@Malacrida1`. References...

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.


For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References