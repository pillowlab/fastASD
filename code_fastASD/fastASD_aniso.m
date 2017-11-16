function [kest,ASDstats,kest_iso,ASDstats_iso] = fastASD_aniso(x,y,dims,minlens,nxcirc)
% Automatic smoothness determination (ASD) for nD filter with diff length scales
%
% [kest,ASDstats,kest_iso,ASDstats_iso] = fastASD_aniso(x,y,dims,minlens,nxcirc)
%
% Empirical Bayes estimate of regression coefficients under ASD (squared exponential) prior.
% Uses Fourier representation of ASD covariance matrix so prior covariance is diagonal.
%
% 
% INPUT:
% -----
%        x [nt x nx] - stimulus matrix (each row contains all spatial pixels at a single time)
%         y [nt x 1] - response vector
%       dims [2 x 1] - dims for stimulus, specifying how to reshape each x row to an image
%    minlens [1 x 1] or [m x 1] - minimum length scale for each dimension (can be scalar)
%     nxcirc [2 x 1] - circular boundary for each dimension (OPTIONAL)
%
% OUTPUT:
% ------
%   kest     [m x 1] - ASD estimate of regression weights under anisotropic ASD prior
%           ASDstats - struct with fitted hyperparameters, Hessian, posterior
%                      confidence intervals and covariance matrix
%   kest_iso [m x 1] - ASD estimate with isotropic prior (single length scale)
%       ASDstats_iso - struct for isotropic estimate
% 
% Dependencies: calls fastASD to compute isotropic as initializer


CONDTHRESH = 1e8;  % threshold for small eigenvalues

% Parse inputs 
if (nargin < 5)
    nxcirc = ceil(max([dims(:)'+minlens(:)'*4;dims(:)'*1.25]))';
end

nkd = length(dims); % number of distinct length scales (one per dimension).
if nkd==1
    error('no need to compute anisotropic ASD estimate: only 1 dimension');
end


% Compute isotropic ASD estimate
[kest_iso,ASDstats_iso,dd] = fastASD(x,y,dims,minlens,nxcirc);


% ========== Compute anisotropic estimate ============

% Make handle for loss function, set initial value and bounds
lfun = @(prs)neglogev_ASDspectral_nD(prs,dd,dd.wwnrm,CONDTHRESH);  % loss function
hprs_init = dd.hprs([ones(nkd,1);2;3]); % initial params from isotropic case
LB = [minlens(:); 1e-2; 1e-2];  % lower bounds
UB = inf(nkd+2,1); % upper bounds
fminopts = optimset('gradobj','on','Hessian','on','display','off',...
    'algorithm','trust-region-reflective','maxfunevals',1000,'maxiter',1000);

% HessCheck(lfun,hprs_init);  % check grad & Hessian numerically
hprshat = fmincon(lfun,hprs_init,[],[],[],[],LB,UB,[],fminopts); % run optimization

% Compute Hessian and Fourier-domain support at posterior mode
[neglogEv,~,H,muFFT,LpostFFT,ii] = lfun(hprshat);

% Transform estimate back to space-time domain
Binds = false(size(dd.Bii)); % initialize
Binds(dd.Bii) = ii; % indices in basis
kest = kronmult(dd.Bfft,muFFT,Binds); % project to space-time basis by inverse DFT

% Report rank of prior at termination
fprintf('     full nd-ASD: terminated with rank of Cprior = %d\n',sum(ii));

% Check if length scale is at minimum allowed range
if any (hprshat(1:end-2) <= (minlens+.001))
    fprintf('\n');
    warning(['Solution is at minimum length scale\n',...
        'Consider re-running with shorter ''minlen'' \n\n']);
end

% Assemble output statistics
if nargout > 1
    lhat = hprshat(1:end-2); % length scale
    trhohat = hprshat(end-1); % transformed rho param
    rhohat = trhohat/(prod(lhat)*(2*pi)^(nkd/2)); % original rho para
    Ajcb = [eye(nkd), zeros(nkd,2);
        (2*pi)^(nkd/2)*[lhat'*rhohat, prod(lhat)], 0; ... % Jacobian of param remapping
        zeros(1,nkd+1), 1];
    
    ASDstats.rho = rhohat;  % rho hyperparameter
    ASDstats.len = lhat;  % length scale hyperparameter
    ASDstats.nsevar = hprshat(end); % noise variance
    ASDstats.H = Ajcb'*H*Ajcb;  % Hessian of hyperparameters % FIX
    ASDstats.ci = sqrt(diag(inv(ASDstats.H))); % 1SD posterior CI for hyperparams
    ASDstats.neglogEv = neglogEv; % negative log-evidence at solution
    
    % Compute diagonal of posterior covariance over filter coeffs
    ASDstats.Lpostdiag = kroncovdiag(dd.Bfft,LpostFFT,Binds);
   
end
