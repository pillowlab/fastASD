function [khat,hprs] = autoRidgeRegress_fp(dstruct,lam0,opts)
% "Automatic" ridge regression w/ fixed-point evidence optimization for hyperparams 
%  
% [khat,hprs] = autoRidgeRegression_fp(datastruct,lam0,opts)
%
% Computes maximum marginal likelihood estimate for prior variance of an
% isotropic Gaussian prior (1/alpha) and variance of additive noise (nsevar)
%
% Note: traditional ridge parameter equals [prior precision] * [noise variance], 
%       i.e., lambda = (hprs.alpha * hprs.nsevar)
%
% INPUT:
% -------
%  dstruct - data structure with fields:
%            .xx - stimulus autocovariance matrix X'*X
%            .xy - stimulus-response cross-covariance X'*Y
%            .yy - response variance Y'*Y
%            .ny - number of samples 
%   lam0 - initial value of the ratio of ridge parameter (OPTIONAL)
%            (equal to noise variance divided by prior variance)
%   opts - options stucture w fields  'maxiter' and 'tol' (OPTIONAL)
%
%
% OUTPUT:
% -------
%     khat - "empirical bayes' estimate of kernel k
%     hprs - struct with fitted hyperparameters 'alpha' and 'nsevar'
%
%  Updated 2015.03.24 (jwp)

MAXALPHA = 1e6; % Maximum allowed value for prior precision

% Check input arguments
if nargin < 2
    lam0 = 10;
end
if nargin < 3
    opts.maxiter = 100;
    opts.tol = 1e-6;
end

% ----- Initialize some stuff -------
jcount = 1;  % counter
dparams = inf;  % Change in params from previous step
xx = dstruct.xx; % extract
xy = dstruct.xy;
yy = dstruct.yy;
ny = dstruct.ny;

nx= size(xx,1); % number of stimulus dimnesions
Lmat = speye(nx);  % Diagonal matrix for prior

% ------ Initialize alpha & nsevar using MAP estimate around lam0 ------
kmap0 = (xx + lam0*Lmat)\xy;  % MAP estimate given lam0
nsevar = yy - 2*kmap0'*xy + kmap0'*xx*kmap0; % 1st estimate for nsevar: var(y-x*kmap0); 
alpha = lam0/nsevar;

% ------ Run fixed-point algorithm  ------------
while (jcount <= opts.maxiter) && (dparams>opts.tol) && (alpha <= MAXALPHA)
    CpriorInv = Lmat*alpha;
    
    Cpost = inv(xx/nsevar + CpriorInv);  % posterior covariance
    mupost = (Cpost*xy)/nsevar; % posterior mean
    alphanew = (nx- alpha.*trace(Cpost))./sum(mupost.^2); % update for alpha
    
    numerator = yy - 2*mupost'*xy + mupost'*xx*mupost;
    nsevarnew = sum(numerator)./(ny-sum(1-alpha*diag(Cpost)));

    % update counter, alpha & nsevar
    dparams = norm([alphanew;nsevarnew]-[alpha;nsevar]);
    jcount = jcount+1;
    alpha = alphanew;
    nsevar = nsevarnew;
end
khat = (xx + alpha*nsevar*Lmat)\(xy); % final estimate (posterior mean after last update)

if alpha >= MAXALPHA
    fprintf(1, 'Finished autoRidgeRegression: filter is all-zeros\n');
    khat = mupost*0;  % Prior variance is delta function
elseif jcount < opts.maxiter
    fprintf('Finished autoRidgeRegression in #%d steps\n', jcount);
else
    fprintf(1, 'Stopped autoRidgeRegression: MAXITER (%d) steps; dparams last step: %f\n', ...
        jcount, dparams);
end

% Put hyperparameter estimates into struct
hprs.alpha = alpha;
hprs.nsevar = nsevar;