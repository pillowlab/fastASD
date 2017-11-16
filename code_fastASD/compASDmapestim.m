function kest = fastASD(x,y,dims,asdstats,dd)
% Compute MAP estimate for linear filter with fixed ASD hyperparameters
%
% kest = fastASD(x,y,asdstats,ddstruct)
%
% Assumes x and y have same size as originally used to construct structs
% asdstats and dd.
%
% 
% INPUT:
% -----
%        x [nt x nx] - stimulus matrix; each row contains is single vector of regressors
%         y [nt x 1] - response vector
%           ASDstats - struct with fitted ASD hyperparams
%                 dd - struct with info about dimensions & support in Fourier basis 
%
% OUTPUT:
% ------
%    kest [m x 1] - ASD estimate of regression weights under isotropic ASD prior


% Compute sufficient stats
Bx = kronmulttrp(dd.Bfft,x'); % convert to Fourier domain
Bx = Bx(dd.Bii,:); % prune unneeded freqs
xx = Bx*Bx'; % compute stimulus covariance
xy = Bx*y; % compute stim-response cross-covariance

% Compute prior cov
kest_dft = (xx(dd.ii,dd.ii) + asdstats.nsevar*dd.Cinv)\xy(dd.ii);
Binds = false(size(dd.Bii)); % initialize
Binds(dd.Bii) = dd.ii; % indices in basis
kest = kronmult(dd.Bfft,kest_dft,Binds); % take inverse Fourier transform
kest = reshape(kest,dims); % reshape


