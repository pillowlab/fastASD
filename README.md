# fastASD
Fast inference for high-D receptive fields using Automatic Smoothness
Determination (ASD) in Matlab

- general method for regression problems with smooth weights

**Description:** Performs evidence optimization for hyperparameters in
  a Gaussian linear regression model with a Gaussian Process (GP)
  prior on the regression weights, and returns *maximum a posteriori*
  (MAP) estimate of the weights given optimal
  hyperparameters. 

### Original reference: ###
[Sahani, M. & Linden, J. Evidence optimization techniques for estimating stimulus-response functions. *NIPS* (2003)](http://www.gatsby.ucl.ac.uk/~maneesh/papers/nips02-evidence.pdf).

### Reference for this implementation ###
[Aoi MC & Pillow JW.  Scalable Bayesian inference for high-dimensional neural receptive fields. *bioRxiv* (2017)](https://www.biorxiv.org/content/early/2017/11/01/212217) 


Download
==========

* **Download**:   zipped archive  [LNPfitting-master.zip](https://github.com/pillowlab/LNPfitting/archive/master.zip)
* **Clone**: clone the repository from github: ```git clone git@github.com:pillowlab/LNPfitting.git```

Getting started
===========

* Launch matlab and cd into the directory containing the code   
(e.g. `cd fastASD/`).

* Run script "setpaths.m" to set local paths:  
`> setpaths`

* cd to directory demos:  
`> cd demos`

* **Examine the demo scripts for example application to simulated data:** 
	*  `test_fastASD_1D.m` - illustrate estimation of a 1D
       (e.g. purely temporal) receptive field using fastASD.
	*  `test_fastASD_2D.m` - for 2D (e.g. 1D space x time ) receptive
    field with same length scale (i.e. smoothness level) along each axis.
	* `test_fastASD_2Dnonisotropic.m` - for 2D receptive field, but
    with different length scale (smoothness) along each axis.
	* `test_fastASD_3D.m` - for 3D (e.g. 2D space x time) receptive field.


* **More advanced demo scripts:**

	* `test_fastASD_1Dgroupedcoeffs.m` - model with multiple vectors of
  regression weights for different covariates, each of which should be smooth (e.g. 1 
  filter governing stimulus and 1 governing running speed). 

	* `test_fastASD_1Dnonuniform.m` - run ASD for weight vectors which
      are not evenly spaced in time (or space),

	* `test_fastASD_2Dnonunif.m` - same but for non-uniformly sampled
      2D data,  e.g. for inferring a rat   hippocampal place fields
      from location data that does not sit on a 2D  grid.  (See, e.g.,
      [Muragan et al
      2017](https://www.biorxiv.org/content/early/2017/06/26/155929?rss=1) )


Notes
=====

- The ASD model includes three hyperparameters (for 1D or isotropic
prior):  
  * `rho` - prior variance of the regression weights
  * `len` - length scale, governing smoothness (larger => smoother).
  * `sig^2` - variance of the observation noise (from Gaussian
  likelihood term).

- The only change for non-isotropic version is to have a different
length scale governing smoothness in each direction.


- code returns MAP estimate of RF, posterior confidence intervals of
the RF, and confidence intervals on hyperparameters based on the
inverse Hessian of the marginal likelihood.

- Computational efficiency of this implementation comes from using a
  spectral (Fourier basis) representation of the ASD prior, which is also known as
  Gaussian, or Radial Basis Function (RBF), or "squared exponential"
  covariance function.

