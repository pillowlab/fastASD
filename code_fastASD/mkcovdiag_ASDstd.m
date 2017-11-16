function [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = mkcovdiag_ASDstd(len,trho,wwnrm)
% Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
%
% [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = mkcovdiag_ASDstd(trhos,lens,wwnrm)
%
% Compute discrete ASD (RBF kernel) eigenspectrum in Fourier domain, and
% its derivatives w.r.t. to the model parameters trho and len
%
% INPUT:
%         len [1 x 1] or [n x 1] - ASD length scales
%        trho [1 x 1] or [n x 1] - Fourier domain prior variance
%       wwnrm [n x 1] - vector of squared normalized Fourier frequencies
%        
% OUTPUT:
%      cdiag [n x 1] - vector of eigenvalues of C for frequencies in w
%     mdcinv [n x 2] - 1st derivs [dC^-1/dlen, dC^-1/drho]
%        mdc [n x 2] - 1st derivs [dC/dlen, dC/drho]
%    mddcinv [n x 3] - 2nd derivs of C^-1 w.r.t [dlen^2, dlen*drho,drho^2]


% Compute diagonal of ASD covariance matrix
cdiag = trho.*exp(-.5*wwnrm.*len.^2);

% 1st derivative of inv(Cdiag)
if nargout > 1
    dcinvdthet = [(len.*wwnrm)./cdiag, ...  % dC^-1/dl
        -(1./trho)./cdiag];        % dC^-1/drho
end

% 1st derivative of Cdiag 
if nargout > 2
    dcdthet = [-len.*wwnrm.*cdiag, ...   % dC/dl
        (cdiag./trho)]; % dC/drho
        
end

% 2nd derivative of inv(Cdiag)
if nargout > 3
    ddcinvdthet = [(wwnrm + len.^2.*wwnrm.^2)./cdiag, ... % d^2 C^-1 /dl^2
        -(1./trho).*(len.*wwnrm)./cdiag, ...   % d^2 C^-1 /drho dl
        (2./(trho.^2.*cdiag))]; % d^2 C^-1 /drho^2
end
