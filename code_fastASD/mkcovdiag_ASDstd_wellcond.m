function cdiag = mkcovdiag_ASDstd_wellcond(len,trho,wwnrm,adddiag)
% Eigenvalues of ASD covariance (as diagonalized in Fourier domain),
% well-conditioned by adding a constant offset
%
% [cdiag] = mkcovdiag_ASDstd(trhos,lens,wwnrm,dcadd)
%
% Compute discrete ASD (RBF kernel) eigenspectrum in Fourier domain with
% lower bound
%
% INPUT:
%         len [1 x 1] or [n x 1] - ASD length scales
%        trho [1 x 1] or [n x 1] - Fourier domain prior variance
%       wwnrm [n x 1] - vector of squared normalized Fourier frequencies
%     adddiag [1 x 1] - additive constant to keep eigenspectrum well
%                       conditioned (DEFAULT = 10e-08)
%        
% OUTPUT:
%      cdiag [n x 1] - vector of eigenvalues of C for frequencies in w

if nargin < 4
    adddiag = 1e-08;
end

% Compute diagonal of ASD covariance matrix
cdiag = trho.*exp(-.5*wwnrm.*len.^2)+adddiag;
