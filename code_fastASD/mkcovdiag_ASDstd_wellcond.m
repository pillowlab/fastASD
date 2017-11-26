function [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = mkcovdiag_ASDstd_wellcond(len,trho,wwnrm,adddiag)
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
%                       conditioned (DEFAULT = 10e-06)
%        
% OUTPUT:
%      cdiag [n x 1] - vector of eigenvalues of C for frequencies in w

if nargin < 4
    adddiag = 1e-6;
end

% Compute diagonal of ASD covariance matrix
cdiag0 = trho.*exp(-.5*wwnrm.*len.^2);
cdiag = cdiag0+adddiag;

% 1st derivative of inv(Cdiag)
if nargout > 1
    dcinvdthet = [(len.*wwnrm).*(cdiag0./cdiag.^2), ...  % dC^-1/dl
        -(1./trho).*(cdiag0./cdiag.^2)];        % dC^-1/drho
end

% 1st derivative of Cdiag 
if nargout > 2
    dcdthet = [-len.*wwnrm.*cdiag0, ...   % dC/dl
        (cdiag0./trho)]; % dC/drho
        
end

% 2nd derivative of inv(Cdiag)
if nargout > 3
    ddCdl2 = (-wwnrm+len.^2*wwnrm.^2).*cdiag0; % 2nd deriv of C w.r.t. l
    ddCdlr = dcdthet(:,1)./trho; % 2nd deriv of C w.r.t. l by rho
    ddcinvdthet = ...
        [(-cdiag.*ddCdl2 + 2*dcdthet(:,1).^2)./cdiag.^3, ...% d^2 C^-1 /dl^2
        (-cdiag.*ddCdlr + 2*dcdthet(:,1).*dcdthet(:,2))./cdiag.^3, ... d^2 C^-1 / dl dr
        2*dcdthet(:,2).^2./cdiag.^3]; % d^2 C^-1 dr

end
