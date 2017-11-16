function C = mkcov_ASD(len,rho,nx)
% Generate ASD covariance matrix, also known as RBF or squared-exponential kernel
%
% C = mkCov_ASD(len,rho,nx)
%
% Covariance matrix parametrized as:
%  C_ij = rho*exp(((i-j)^2/(2*len^2))
%
% INPUTS:
%     len - length scale of ASD kernel (determines smoothness)
%     rho - maximal prior variance ("overall scale")
%      nx - number of indices (sidelength of covariance matrix)
%
% OUTPUT:
%   C [nx x nx] - covariance matrix
%
% Updated 2014.11.27 (jwp)

ix = (1:nx)'; % indices for coefficients
sqrdists = bsxfun(@minus,ix,ix').^2; % squared distances between indices
C = rho*exp(-.5*sqrdists/len.^2); % the covariance matrix
    