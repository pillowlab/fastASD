function [cdiag,dcinv,dc,ddcinv] = mkcovdiag_ASD(len,rho,nxcirc,wvecsq)
% Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
%
% [cdiag,dcdiag,ddcdiag] = mkcovdiag_ASD(rho,l,nxcirc,wvecsq)
%
% Compute discrete ASD (RBF kernel) eigenspectrum using frequencies in [0, nxcirc].
% See mkCov_ASD_factored for more info
%
% INPUT:
%         len - length scale of ASD kernel (determines smoothness)
%         rho - maximal prior variance ("overall scale")
%      nxcirc - number of coefficients to consider for circular boundary 
%        wvecsq - vector of squared frequencies for DFT 
%        
% OUTPUT:
%      cdiag [nxcirc x 1] - vector of eigenvalues of C for frequencies in w
%     dcinv [nxcirc x 2] - 1st derivs [dC^-1/drho, dC^-1/dlen]
%        dc [nxcirc x 2] - 1st derivs [dC / drho , dC / dlen]
%    ddcinv [nxcirc x 3] - 2nd derivs of C-1 w.r.t [drho^2, drho*dlen, dlen^2]
%
% Note: nxcirc = nx corresponds to having a circular boundary 


% Compute diagonal of ASD covariance matrix
const = (2*pi/nxcirc)^2; % constant 
ww = wvecsq*const;  % effective frequency vector
cdiag = sqrt(2*pi)*rho*len*exp(-.5*ww*len^2);

% 1st derivative of inv(Cdiag)
if nargout > 1
    nw = length(wvecsq);
    dcinv = [(-1/len + len*ww)./cdiag, ...  % dC^-1/dl
        -(1/rho)./cdiag];                   % dC^-1/drho
end

% 1st derivative of Cdiag 
if nargout > 2
    dc = [(1/len - len*ww).*cdiag(:,[1 1]), ... % dC/dl
        (1./rho)*ones(nw,1)];                   % dC/drho
end

% 2nd derivative of inv(Cdiag)
if nargout > 3
    ddcinv = [(2/len^2 - ww + len^2*ww.^2), ... % d^2 C^-1 /dl^2
        (1/rho)*(1/len - len*ww), ...  % d^2 C^-1 /drho dl
        (2/rho^2)*ones(nw,1)] ...;     % d^2 C^-1 /drho^2
        ./ cdiag(:,ones(1,3));
end
