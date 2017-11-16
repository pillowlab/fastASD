function [kest,ASDstats] = fastASDnu(x,y,xpos,minlens,nxcirc)
% Fast automatic smoothness determination (ASD) for non-uniformly spaced weights
%
% [kest,ASDstats] = fastASD_2Dnu(x,y,xpos,minlen,nxcirc)
%
% Empirical Bayes estimate of regression coefficients under smoothing
% ("squared exponential") prior, with maximum marginal likelihood estimate
% of prior covariance parameters.
%
% Implementation: uses Fourier representation of stimuli and ASD covariance
% matrix, so prior is diagonal.  
% 
% INPUT:
% -------
%        x [nt x nx] - stimulus matrix (each row contains spatial stimulus at single time)
%        y [nt x 1 ] - response vector
%     xloc [nt x 1 ] - spatial location for each regression weight
%   minlen [1 x 1] - minimum length scale (the larger, the faster)
%   nxcirc [1 x 1] - circular boundary  (OPTIONAL)
%
% OUTPUT:
% --------
%   kest [nk x 1] - ASD estimate of regression weights
%   ASDstats - struct with fitted hyperparameters, Hessian, posterior covariance
%              and (finallY) kest sampled on a grid (if desired)


%% ========= Parse inputs and initialize hyperparams =====================

CONDTHRESH = 1e8;  % threshold for small eigenvalues
if nargin < 5
    nxcirc = [];
end
nkd = size(xpos,2); % number of filter dimension

% Determine size of matrix of regressors 
nsamps = size(x,1);

% Compute sufficient statistics in Fourier domean
[dd,wwnrm,Bfft] = compLSsuffstats_fourier_nu(x,y,xpos,minlens,nxcirc,CONDTHRESH);
wwnrmtot = sum(wwnrm,2);

% -- initialize range for hyperparameters -----
% This is just for the grid search to initialize numerical search. Ranges
% are based on crude heuristics, so definitely room for improvement here.

% lengthscale range
lrange = [min(minlens),max(max(minlens)*nkd,min(range(xpos))/2)];

% Rho range
rhomax = 2*(dd.yy./nsamps)/mean(diag(dd.xx)); % ratio of variance of output to intput
rhomin = min(1,.1*rhomax);  % minimum to explore
rhorange = [rhomin,rhomax];

% noise variance sigma_n^2
nsevarmax = dd.yy/nsamps; % marginal variance of y
nsevarmin = min(1,nsevarmax*.01); % var ridge regression residuals
nsevarrange = [nsevarmin, nsevarmax];


% Change of variables to tilde rho (which separates rho and length scale)
trhorange = (2*pi)^(nkd/2)*rhorange.*[min(lrange), max(lrange)].^nkd;


%% ========= Grid search for initial hyperparameters =============================

% Make handle for loss function
lfun = @(prs)neglogev_ASDspectral(prs,dd,wwnrmtot,CONDTHRESH);  % loss function

ngrid = 4;  % search a 4 x 4 x 4 grid for initial value of hyperparams
rnges = [lrange;trhorange;nsevarrange];
[nllvals,gridpts] = grideval(ngrid,rnges,lfun);
[hprs0,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2),gridpts(:,3)); % find minimum


%% ========== Optimize evidence using fmincon ======================================

LB = [min(minlens);1e-2; 1e-2];  % lower bounds
UB = inf(3,1); % upper bounds
fminopts = optimset('gradobj','on','Hessian','on','display','off','algorithm','trust-region-reflective');
% HessCheck(lfun,hprs0);  % check gradient and Hessian numerically, if desired

hprshat = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts); % run optimization


%% =====  compute posterior mean and covariance at maximizer of hyperparams =========

% Compute Hessian and Fourier-domain support at posterior mode
[neglogEv,~,H,muFFT,LpostFFT,ii] = lfun(hprshat);
kest = Bfft(ii,:)'*muFFT;  % inverse Fourier transform of Fourier-domain mean

% Report rank of prior at termination
fprintf('isotropic nd-ASD: terminated with rank of Cprior = %d\n',sum(ii));

% Check if length scale is at minimum allowed range
if hprshat(1) <= min(minlens)+.001
    fprintf('\n');
    warning(['Solution is at minimum length scale\n',...
        'Consider re-running with shorter ''minlen'' \n\n']);
end

% Assemble summary statistics for output 
if nargout > 1
    lhat = hprshat(1); % length scale
    trhohat = hprshat(2); % transformed rho param
    rhohat = trhohat/(lhat*sqrt(2*pi)).^(nkd); % original rho param
    Ajcb = [1 0 0;  % Jacobian of param remapping from trho to rho
        (2*pi).^(nkd/2)*[nkd*lhat^(nkd-1)*rhohat, lhat^nkd], 0;
        0 0 1]; 
    
    ASDstats.len = lhat;  % length scale hyperparameter
    ASDstats.rho = rhohat;  % rho hyperparameter
    ASDstats.nsevar = hprshat(end); % noise variance
    ASDstats.H = Ajcb'*H*Ajcb;  % Hessian of hyperparameters
    ASDstats.ci = sqrt(diag(inv(ASDstats.H))); % 1SD posterior CI for hyperparams
    ASDstats.neglogEv = neglogEv; % negative log-evidence at solution

    % Just compute diagonal of posterior covariance
    ASDstats.Lpostdiag = sum(Bfft(ii,:).*(LpostFFT*Bfft(ii,:)))';
    
    % % If full posterior cov for filter is desired:
    % ASDstats.Lpost = Bfft(ii,:)'*LpostFFT*Bfft(ii,:);

end
