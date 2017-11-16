function [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = mkcovdiag_ASDstd_nD(trho,lens,wwnrm)
% Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
%
% [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = mkcovdiag_ASDstd(trhos,lens,wwnrm)
%
% Compute discrete ASD (RBF kernel) eigenspectrum in Fourier domain, and
% its derivatives w.r.t. to the model parameters trho and len
%
% INPUT:
%         rho [1 x 1] or [n x 1] - maximal of Fourier domain prior variance
%        lens [1 x 1] or [n x 1] -  ASD length scales
%      nxcirc [1 x 1] or [n x 1] - number for circular boundary for each coeff
%      wvecsq [n x 1] - vector of squared normalized Fourier frequencies
%        
% OUTPUT:
%      cdiag [n x 1] - vector of eigenvalues of C for frequencies in w
%     mdcinv [n x 2] - 1st derivs [dC^-1/drho, dC^-1/dlen]
%        mdc [n x 2] - 1st derivs [dC/drho, dC/dlen]
%    mddcinv [n x 3] - 2nd derivs of C^-1 w.r.t [drho^2, drho*dlen,dlen^2]


% Compute diagonal of ASD covariance matrix
wwtot = wwnrm*(lens.^2);  % summed squared frequencies
wwnrmlen = bsxfun(@times,wwnrm,lens'); % frequency times length scale
cdiag = trho.*exp(-.5*wwtot);

% 1st derivative of inv(Cdiag)
if nargout > 1
    dcinvdthet = [bsxfun(@rdivide,wwnrmlen,cdiag), ... % dC^-1/dl_i
        -(1./trho)./cdiag];                            % dC^-1/drho
end

% 1st derivative of Cdiag 
if nargout > 2
    dcdthet = [-bsxfun(@times,wwnrmlen,cdiag), ...   % dC/dl
        (cdiag./trho)];                              % dC/drho
end

% 2nd derivative of inv(Cdiag)
if nargout > 3

    % Do some index magic
    [ri,ci] = triuinds(length(lens));  % get indices for upper triangle
    iidiag = (ri==ci);                 % indices for diagonal elements
    
    % Compute d^2 C^-1  / dli dlj terms
    ddCinvdlens = wwnrmlen(:,ri).*wwnrmlen(:,ci);
    ddCinvdlens(:,iidiag) = ddCinvdlens(:,iidiag)+wwnrm;
    ddCinvdlens = bsxfun(@rdivide,ddCinvdlens,cdiag);
    
    % d^2 C^-1/ dl drho terms
    ddCinvdlendrho = -(1./trho).*(bsxfun(@rdivide,wwnrmlen,cdiag));
    
    % d^2 C^-1 / drho^2 term
    ddCinvdrho2 = 2./(trho.^2.*cdiag);
    
    % Assemble terms
    ddcinvdthet = [ddCinvdlens, ddCinvdlendrho, ddCinvdrho2];

end

% 
% % SAFE TEST CODE
% 
% qrho = 2./(trho.^2.*cdiag);
% qrhol1 = -(lens(1)./trho)*(wwnrm(:,1)./cdiag);
% ql1l1 = (wwnrm(:,1)+wwnrm(:,1).^2*lens(1)^2)./cdiag;
% 
% keyboard;
% ddcinvdthet(:,1:3) = [qrho,qrhol1,ql1l1];
%ddcinvdthet = [qrho,qrhol1,ql1l1];