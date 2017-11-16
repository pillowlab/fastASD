function nle = negLogEv_genericdualform(X,Y,nsevar,Cprior)
% function nle = negLogEv_genericdualform(X,Y,nsevar,C)
%
% Computes negative log of the evidence (marginal likelihood) for linear Gaussian
% regression, given regressors X, observations Y, noisevariance nsevar, and prior
% covariance Cprior.
%
% INPUTS:
%        X [T x nx] - regressors (each row holds regressors for single observation)
%        Y [T x 1] - observations
%           nsevar - variance of observation noise
%   Cprior [nx x nx] - prior covariance over (assumed zero mean) regression weights
%
% OUTPUT: 
%   nle - negative loglikelihood, computed in dual space representation
%
% Note: requires that number of observations T is not so big that we can't hold a T x T 
% matrix in memory  


ny = length(Y);

nle = -logmvnpdf(Y',zeros(1,ny),X*Cprior*X'+nsevar*diag(ones(ny,1)));

