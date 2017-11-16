function [neglogev,grad,H,mupost,Lpost,ii] = neglogev_ASDspectral_group(prs,dd,Bgrp,wwnrm,condthresh)
% Neg log-evidence for ASD regression model with pre-diagonalized inputs
%
% [neglogev,dnegL,ddnegL] = negLogEv_ASDspectral(prs,dd,wvecsq);
%    or
% [neglogev,grad,H] = negLogEv_ASDspectral(prs,dat,wvecsq,opts);
%
% Computes negative log-evidence: 
%    -log P(Y|X,sig^2,C) 
% under linear-Gaussian model: 
%       y = x'*w + n,   n ~ N(0,sig^2)
%       w ~ N(0,C)
% where C is ASD (or RBF or "squared exponential") covariance matrix
% 
% INPUTS:
% -------
%  prs [2*ngrp + 1 x 1] - ASD parameters [rho (marginal var); len (length); nsevar].
%          dd - data structure with fields:
%         .xx - stimulus autocovariance matrix X'*X in Fourier domain
%         .xy - stimulus-response cross-covariance X'*Y in Fourier domain
%         .yy - response variance Y'*Y
%         .nsamps - number of samples 
%        Bgrp - sparse matrix mapping the rho and len hyperparam vectors to coeffs
%      wvecsq - vector of squared Fourier frequencies.
%  condthresh - threshold for condition number of K (Default = 1e8).
%
% OUTPUT:
% -------
%   neglogev - negative marginal likelihood
%   grad - gradient
%   H - Hessian
%   mupost - mean of posterior over regression weights
%   Lpost - posterior covariance over regression weights
%   ii - logical vector indicating which DFT frequencies are not pruned


ngrp = size(Bgrp,2); % number of groups of coefficients

% Unpack parameters
lens = Bgrp*prs(1:ngrp); % vector of rhos for each coeff
trhos = Bgrp*prs(ngrp+1:2*ngrp);  % vector of lens for each coeff
nsevar = prs(end);

% Find indices for which eigenvalues too small
ii = wwnrm < (2*log(condthresh)./lens.^2);
ni = sum(ii); % number of non-zero DFT coefs / rank of covariance after pruning

% Prune XX and XY Fourier coefficients and divide by nsevar
Bgrpred = Bgrp(ii,:); % reduced group indices matrix
XX = dd.xx(ii,ii)/nsevar; 
XY = dd.xy(ii)/nsevar;

% Build prior covariance matrix from parameters
switch nargout 
    case {0,1} % compute just diagonal of C
        cdiag = mkcovdiag_ASDstd(lens(ii),trhos(ii),wwnrm(ii));
    case 2
        % compute diagonal of C and deriv of C^-1
        [cdiag,dcinv] = mkcovdiag_ASDstd(lens(ii),trhos(ii),wwnrm(ii)); 
    otherwise
        % compute diagonal of C, 1st and 2nd derivs of C^-1 and C
        [cdiag,dcinv,dcdiag,ddcinv] = mkcovdiag_ASDstd(lens(ii),trhos(ii),wwnrm(ii)); 
end

Cinv = spdiags(1./cdiag,0,ni,ni); % inverse cov in diagonalized space

if nargout <=1  
    % compute negative loglikelihood only
    trm1 = -.5*(logdet(XX+Cinv) + sum(log(cdiag)) + (dd.nsamps)*log(2*pi*nsevar)); % Log-determinant term
    trm2 = .5*(-dd.yy/nsevar + XY'*((XX+Cinv)\XY));   % Quadratic term
    neglogev = -trm1-trm2;  % negative log evidence
end

% Compute negative loglikelihood and gradient
if nargout >= 2  
    % Make stuff we'll need
    Lpostinv = (XX+Cinv);  % inverse of posterior covariance
    Lpost = inv(Lpostinv); % posterior covariance
    Lpdiag = diag(Lpost);  % diagonal of posterior cov
    mupost = Lpost*XY;     % posterior mean
    
    % --- Compute neg-logevidence ----
    trm1 = -.5*(logdet(Lpostinv) + sum(log(cdiag)) + (dd.nsamps)*log(2*pi*nsevar));
    trm2 = .5*(-dd.yy/nsevar + XY'*Lpost*XY);    % Quadratic term
    neglogev = -trm1-trm2;  % negative log evidence
    
    % --- Compute gradient ------------
    % Derivs w.r.t hyperparams rho and len
    dLdthet = -.5*Bgrpred'*bsxfun(@times,dcinv,(cdiag - (Lpdiag + mupost.^2)));
    % Deriv w.r.t noise variance 'nsevar'
    RR = .5*(dd.yy/nsevar - 2*mupost'*XY + mupost'*XX*mupost)/nsevar; % Squared Residuals / 2*nsevar^2
    Tracetrm = .5*(ni-dd.nsamps-sum((Lpdiag./cdiag)))/nsevar;
    dLdnsevar = -Tracetrm-RR;

    % Combine them into gardient vector
    grad = [dLdthet(:); dLdnsevar];
    
end

%  Compute Hessian
if nargout >= 3

    % theta terms (rho and len)
    nthet = 2;  % number of theta variables (rho and len)
    vn = ones(1,nthet*ngrp); % vector of 1s of length ntheta
    
    % Make matrix with dcinv for each parameter in a second column
    Mdcinv = [bsxfun(@times,Bgrpred,dcinv(:,1)),bsxfun(@times,Bgrpred,dcinv(:,2))];
    Mdcdiag = [bsxfun(@times,Bgrpred,dcdiag(:,1)),bsxfun(@times,Bgrpred,dcdiag(:,2))];
    Mddcinv = [bsxfun(@times,Bgrpred,ddcinv(:,1)),...
        bsxfun(@times,Bgrpred,ddcinv(:,2)),...
        bsxfun(@times,Bgrpred,ddcinv(:,3))];
    
    % Derivs of posterior covariance diagonal and posterior mean w.r.t theta
    dLpdiag = -(Lpost.^2)*Mdcinv; % Deriv of diag(Lpost) w.r.t thetas
    dmupost = -(Lpost)*bsxfun(@times,Mdcinv,mupost); % Deriv of mupost w.r.t thetas
    [ri,ci] = triuinds(nthet*ngrp);  % get indices for rows and columns of upper triangle
    trm1stuff = -.5*(Mdcdiag - (dLpdiag + 2*dmupost.*mupost(:,vn)));
    ddLddthet_trm1 = sum(trm1stuff(:,ri).*Mdcinv(:,ci),1)';
    ddLddthet_trm2 = -.5*Mddcinv'*(cdiag - (Lpdiag + mupost.^2));
    ddLddthet = unvecSymMtxFromTriu(ddLddthet_trm1); % form hessian

    % Generate indices needed to insert the trm2 (those in upper triangle)
    ii1 = sub2ind(ngrp*nthet*[1 1],1:ngrp,1:ngrp)'; % indices for drho^2
    ii2 = sub2ind(ngrp*nthet*[1 1],1:ngrp,ngrp+(1:ngrp))'; % indices for drho,dlen
    ii3 = sub2ind(ngrp*nthet*[1 1],ngrp+(1:ngrp),ngrp+(1:ngrp))';
    iidep = [ii1;ii2;ii3]; 
    ddLddthet(iidep) = ddLddthet(iidep)+ddLddthet_trm2;
    ddLddthet = vecMtxTriu(ddLddthet);

    % nsevar term
    dLpdiagv = sum(Lpost.*(Lpost*XX),2)/nsevar; % Deriv of diag(Lpost) wr.t nsevar
    dmupostv = -(Lpost*(mupost./cdiag))/nsevar; % Deriv of mupost w.r.t nsevar
    ddLdv = -(dLdnsevar/nsevar - RR/nsevar ...
        - sum(dLpdiagv./cdiag)/(2*nsevar) ...
        +((-XY+XX*mupost)'*dmupostv)/nsevar);  % 2nd deriv w.r.t. nsevar

    % Cross term theta - nsevar
    ddLdthetav = .5*Mdcinv'*(dLpdiagv+2*dmupostv.*mupost);
    
    % assemble Hessian 
    H = unvecSymMtxFromTriu([ddLddthet;ddLdthetav; ddLdv]);
    
end