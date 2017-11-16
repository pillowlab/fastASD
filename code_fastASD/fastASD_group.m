function [kest,ASDstats] = fastASD_group(x,y,nkgrp,minlen,nxcirc)
% Infer independent groups of smooth coefficients using diagonalized ASD prior 
%
% [kest,ASDstats] = fastASD_group(dd,nkgrp,minlen,nxcirc)
%
% Empirical Bayes estimate of regression coefficients under automatic
% smoothness determination (ASD) prior (also known as Gaussian or
% squared-exponential kernel), with maximum marginal likelihood estimate of
% prior covariance parameters.  
%
% Implementation: uses Fourier representation of stimuli and ASD covariance
% matrix, so prior is diagonal.  
% 
% INPUT:
% -------
%  dd - data structure with fields:
%          .xx - stimulus autocovariance matrix X'*X
%          .xy - stimulus-response cross-covariance X'*Y
%          .yy - response variance Y'*Y
%          .ny - number of samples 
%  nkgrp - number of elements in each group (assumed to include all coefficients)
%  minlen - [1 x 1] or [ngrp x 1], minimum length scale (for all or for each group)
%  nxcirc - [ngrp x 1], circular boundary for each dimension (OPTIONAL)
%
%
% OUTPUT:
% --------
%   kest [nk x 1] - ASD estimate of regression weights
%   ASDstats - struct with fitted hyperparameters, Hessian, posterior covariance
%
% Note: doesn't include a DC term, so should be applied when response and
% regressors have been standardized to have mean zero.


%% ========= Parse inputs and determine what hyperparams to initialize ===========================

CONDTHRESH = 1e8;  % threshold for small eigenvalues

% Compute sufficient statistics
dd.xx = x'*x;   % stimulus auto-covariance
dd.xy = (x'*y); % stimulus-response cross-covariance
dd.yy = y'*y;   % marginal response variance
[dd.nsamps,nx] = size(x);   % total number of samples and number of coeffs
ngrp = length(nkgrp); % number of groups

% Check to make sure same # of elements in xx as group indices
if sum(nkgrp)~=nx 
    error('Stimulus size dd.xx doesn''t match number of indices in grpid');
end

% Replicate minlen to vector, if needed
if length(minlen)==1
    minlen = repmat(minlen,ngrp,1);
end

% Set circular boundary for each group of coeffs
if nargin < 5
    nxcirc = zeros(ngrp,1); % initialize opts struct
    for jj = 1:ngrp
        nxcircMAX = 1.25*nkgrp(jj);  % maximum based on number of coeffs in group
        nxcirc(jj) = ceil(max(nkgrp(jj)+2*minlen(jj),nxcircMAX)); % set based on maximal smoothness
    end
end

% -- initialize range for hyperparameters -----

% lengthscale range
maxlens = max([minlen'*2;nkgrp(:)'/4])';
lrange = [min(minlen),max(maxlens)];

% Rho range
rhomax = 2*(dd.yy./dd.nsamps)/mean(diag(dd.xx)); % ratio of variance of output to intput
rhomin = min(1,.1*rhomax);  % minimum to explore
rhorange = [rhomin,rhomax];

% noise variance sigma_n^2
nsevarmax = dd.yy/dd.nsamps; % marginal variance of y
nsevarmin = min(1,nsevarmax*.01); % var ridge regression residuals
nsevarrange = [nsevarmin, nsevarmax];

% Change of variables to tilde rho (which separates rho and length scale)
trhorange = sqrt(2*pi)*rhorange.*[min(lrange), max(lrange)];


%% ========= Diagonalize by converting to FFT basis  ==========

opts = struct('nxcirc',nxcirc,'condthresh',CONDTHRESH);
opt1 = opts;

% Generate Fourier basis for each group of coeffs
Bmats = cell(ngrp,1); % Fourier basis for each group
wvecspergrp = cell(ngrp,1); % frequency vector for each group
for jj = 1:ngrp;
    opt1.nxcirc = opts.nxcirc(jj); % pass in ju
    [~,Bmats{jj},wvecspergrp{jj}] = mkcov_ASDfactored([minlen(jj);1],nkgrp(jj),opt1);
end
Bfft = blkdiag(Bmats{:}); % Fourier basis matrices assembled into blkdiag
wvec = cell2mat(wvecspergrp); % Group Fourier frequencies assembled into one vec
nwvec = cellfun(@length,wvecspergrp);
dd.xx = Bfft'*dd.xx*Bfft;  % project xx into Fourier basis for each group of coeffs
dd.xy = Bfft'*dd.xy;    % project xy into Fourier basis for each group

% Make matrix for mapping hyperparams to Fourier coefficients for each group
grouppid = arrayfun(@(n)sparse(ones(n,1)),nwvec,'uniformoutput',0); 
Bgrp = logical(blkdiag(grouppid{:}));

% Compute vector of normalized squared Fourier frequencies
wwnrm = (2*pi./(Bgrp*nxcirc)).^2.*(wvec.^2);  % compute normalized DFT frequencies squared


%% ========= Grid search for initial hyperparameters =============================
% Set loss function for grid search
iigrp = [ones(ngrp,1);2*ones(ngrp,1);3]; % indices for setting group params
lfun0 = @(prs)neglogev_ASDspectral_group(prs(iigrp),dd,Bgrp,wwnrm,CONDTHRESH);  % loss function

% Set up grid
ngrid = 4;  % search a 4 x 4 x 4 grid for initial value of hyperparams
rnges = [lrange;trhorange;nsevarrange];


% Do grid search and find minimum
[nllvals,gridpts] = grideval(ngrid,rnges,lfun0);
[hprs00,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2),gridpts(:,3)); % find minimum
hprs0 = hprs00(iigrp); % initialize hyperparams for each group


%% ========== Optimize evidence using fmincon ======================================
lfun = @(prs)neglogev_ASDspectral_group(prs,dd,Bgrp,wwnrm,CONDTHRESH);  % loss function
LB = [minlen(:); 1e-2*ones(ngrp+1,1)]; % lower bounds
UB = inf(ngrp*2+1,1); % upper bounds

fminopts = optimset('gradobj','on','Hessian','off','display','off','algorithm','trust-region-reflective');
% HessCheck(lfun,hprs0);  % check gradient and Hessian numerically, if desired

hprshat = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts); % run optimization


%% =====  compute posterior mean and covariance at maximizer of hyperparams =========
[neglogEv,~,H,muFFT,LpostFFT,ii] = lfun(hprshat);
kest = Bfft(:,ii)*muFFT;  % inverse Fourier transform of Fourier-domain mean

% Report rank of prior at termination
fprintf('fastASD_group: terminated with rank of Cprior = %d\n',sum(ii));

% Check if length scale is at minimum allowed range
if any(hprshat(1:ngrp) <= minlen(:)+.001)
    fprintf('\n');
    warning(['Solution is at minimum length scale\n',...
        'Consider re-running with shorter ''minlen'' \n\n']);
end

% Assemble summary statistics for output 
if nargout > 1
    % Transform trho back to standard rho
    lhat = hprshat(1:ngrp); % transformed rho param
    trhohat = hprshat(ngrp+1:2*ngrp); % length scale
    a = sqrt(2*pi);
    rhohat = trhohat./(lhat*a); % original rho param
    Ajcb = [eye(ngrp), zeros(ngrp,ngrp+1); ... % Jacobian for params
        a*[diag(rhohat), diag(lhat)], zeros(ngrp,1); ...
        zeros(1,ngrp*2), 1];

    ASDstats.rho = rhohat;  % rho hyperparameter
    ASDstats.len = lhat;  % length scale hyperparameter
    ASDstats.nsevar = hprshat(end); % noise variance
    ASDstats.H = H;  % Hessian of hyperparameters
    ASDstats.H = Ajcb'*H*Ajcb;  % Hessian of hyperparameters
    ASDstats.ci = sqrt(diag(inv(ASDstats.H))); % 1SD posterior CI for hyperparams
    
    ASDstats.neglogEv = neglogEv; % negative log-evidence at solution
    % Just compute diagonal of posterior covariance
    ASDstats.Lpostdiag = sum(Bfft(:,ii)'.*(LpostFFT*Bfft(:,ii)'))';
    
    % % If full posterior cov for filter is desired:
    % ASDstats.Lpost = Bfft(:,ii)*LpostFFT*Bfft(:,ii)'; 
    
end