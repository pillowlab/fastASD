function [kest,ASDstats,dd] = fastASD(x,y,dims,minlens,nxcirc)
% Automatic smoothness determination (ASD) for nD filter w/ isotropic smoothness
%
% [kest,ASDstats] = fastASD(x,y,dims,minlens,nxcirc)
%
% Empirical Bayes estimate of regression coefficients under ASD (squared exponential) prior.
% Uses Fourier representation of ASD covariance matrix so prior covariance is diagonal.
%
% 
% INPUT:
% -----
%        x [nt x nx] - stimulus matrix; each row contains is single vector of regressors
%         y [nt x 1] - response vector
%       dims [n x 1] - dims for stimulus, specifying how to reshape each x row to an image
%    minlens [1 x 1] or [n x 1] - minimum length scale for each dimension (can be scalar)
%     nxcirc [2 x 1] - circular boundary for each dimension (OPTIONAL)
%
% OUTPUT:
% ------
%    kest [m x 1] - ASD estimate of regression weights under isotropic ASD prior
%        ASDstats - struct with fitted hyperparams, Hessian, conf intervals, etc.
%              dd - struct with sufficient stats needed for evidence optimization
%
% Note: model doesn't include a DC term, so x and y should have zero mean

CONDTHRESH = 1e8;  % threshold for small eigenvalues

% Parse inputs 
if (nargin < 5)
    nxcirc = ceil(max([dims(:)'+minlens(:)'*4;dims(:)'*1.25]))';
end
nkd = length(dims); % number of filter dimension

% compute sufficient statistics, projected into Fourier domain
[dd,wwnrm,Bfft,Bii] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,CONDTHRESH);
wwnrmtot = sum(wwnrm,2);

%% ==== Set range for hyperparameters =================
% (This is just for initial grid search, w/ ranges set using crude heuristics)

% lengthscale range
lrange = [min(minlens),max(max(minlens)*3,max(dims)/4)];

% rho (marginal variance)
rhomax = 2*(dd.yy./dd.nsamps)/mean(diag(dd.xx)); % ratio of output to input variance
rhomin = min(1,.1*rhomax);  % minimum to explore (totally ad hoc)
rhorange = [rhomin,rhomax];

% noise variance sigma_n^2
nsevarmax = dd.yy/dd.nsamps; % marginal variance of y
nsevarmin = min(1,nsevarmax*.01); % var ridge regression residuals
nsevarrange = [nsevarmin, nsevarmax];

% Change of variables to tilde rho (which separates rho and length scale)
trhorange = (2*pi)^(nkd/2)*rhorange.*[min(lrange), max(lrange).^nkd];


%% ========= Grid search for initial hyperparameters =============================

% Make handle for loss function
lfun = @(prs)neglogev_ASDspectral(prs,dd,wwnrmtot,CONDTHRESH);  % loss function

ngrid = 4;  % search a 4 x 4 x 4 grid for initial value of hyperparams
rnges = [lrange;trhorange;nsevarrange];  % range for grid 
[nllvals,gridpts] = grideval(ngrid,rnges,lfun); % evaluate evidence on grid
[hprs0,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2),gridpts(:,3)); % find minimum


%% ========== Optimize evidence using fmincon ======================================

LB = [min(minlens); 1e-2; 1e-2];  % lower bounds
UB = inf(3,1); % upper bounds

fminopts = optimset('gradobj','on','Hessian','on','display','off',...
    'algorithm','trust-region-reflective','maxfunevals',1000,'maxiter',1000);
% HessCheck(lfun,hprs0);  % check gradient and Hessian numerically

% run optimization
hprshat = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts);

% Compute Hessian and Fourier-domain support at posterior mode
[neglogEv,~,H,muFFT,LpostFFT,ii,Cinv] = lfun(hprshat);

% Transform estimate back to space-time domain
Binds = false(size(Bii)); % initialize
Binds(Bii) = ii; % indices in basis
kest = kronmult(Bfft,muFFT,Binds); % project to space-time basis by inverse DFT

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
        
    % Compute diagonal of posterior covariance over filter coeffs
    ASDstats.Lpostdiag = kroncovdiag(Bfft,LpostFFT,Binds);
    
    % % If full posterior cov for filter is desired:
    % Lpost = kronmult(Bfft,kronmult(Bfft,LpostFFT,Binds)',Binds);

end

if nargout == 3
    % add Fourier-transformed sufficient statistics and fitted params
    % to dd struct (for possible use by other funcs)
    dd.wwnrm = wwnrm;
    dd.Bfft = Bfft;
    dd.Bii = Bii;
    dd.hprs = hprshat;
    dd.condthresh = CONDTHRESH;
    dd.Cinv = Cinv;
    dd.ii = ii;
end
