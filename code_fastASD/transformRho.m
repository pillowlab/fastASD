function [transformedrho,J] = transformRho(rho,len,nD,direction)
% [trho,J] = transformRho(rho,len,nD,direction)
%
% Function for transforming from marginal variance rho to alternate
% parametrization "trho", which decoupled rho from dependence on length
% scale in the Fourier domain (forward direction), or back.
%
% Inputs:
%       rho - marginal variance for GP prior (forward direction)
%       len - length scale  
%        nD - number of dimensions of GP
% direction - direction ("+1" for forward, "-1" for backward).
%
% Outputs:
%   trho - transformed marginal variance for ASD prior (forward direction)
%      J - Jacobian for transformation from [len; trho] -> [len; rho]
%          (inverse direction only)

if direction == 1
    % Forward direction (rho to tilde-rho)
    transformedrho = rho.*((len*sqrt(2*pi)).^nD);
    
elseif direction == -1
    % Reverse direction (from tilde-rho to rho)
    
    transformedrho = rho./((len*sqrt(2*pi)).^nD);

    % Jacobian of param remapping from [lengthscales, tilde-rho] to 
    %  [lengthscales, rho]
    
    if nargout > 1
        J = [1 0;
            (2*pi).^(nD/2)*[nD*len^(nD-1)*transformedrho, len^nD]];
    end
    
end