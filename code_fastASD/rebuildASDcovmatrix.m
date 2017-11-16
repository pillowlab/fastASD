function [C,Cinv,Sdiag,Bbasis,Bnull] = rebuildASDcovmatrix(dstruct,condthresh)
% [C,Cinv,Sdiag,Bbasis,Bnull] = rebuildASDcovmatrix(dstruct,condthresh)
%
% Takes parameter struct with optimized hyperparameters from 'fastASD'
% output and constructs ASD prior covariance matrix, inverse covariance,
% and a basis for the null space.
%
% Inputs: 
%   dstruct - param struct with info about optimized basis
%   condthresh - threshold on condition number (OPTIONAL; default = 1e12)
% 
% Output: 
%   C - covariance matrix
%   Cinv - inverse covariance matrix
%   Sdiag - vector of non-degenerate eigenvalues of C
%   Bbasis - basis for non-degenereate subspace of C
%   Bnull - basis for null space of C

% Set condthresh, if necessary
if nargin < 2
    condthresh = 1e12;
end

% Construct inverse covariance matrix from learned ASD params
Bfft = kron(dstruct.Bfft{2},dstruct.Bfft{1});
ncol = cols(Bfft);
inds = false(ncol,1);
inds(dstruct.Bii) = dstruct.ii;

% Construct covariance matrix
cdiag = zeros(ncol,1);
cdiag(inds) = 1./diag(dstruct.Cinv);
C = Bfft*diag(cdiag)*Bfft'; 

% Construct inverse covariance matrix
if nargout > 1
    cdiaginv = zeros(ncol,1);
    cdiaginv(inds) = diag(dstruct.Cinv);
    Cinv = Bfft*diag(cdiaginv)*Bfft';  % inverse covariance matrix
end

% Take SVD of C well-conditioned piece
if nargout > 2
    [UU,Sdiag] = svd(C);
    Sdiag = diag(Sdiag);
    ii = Sdiag>Sdiag(1)/condthresh;
    Sdiag = Sdiag(ii);
    Bbasis = UU(:,ii);
    Bnull = UU(:,~ii);
end
