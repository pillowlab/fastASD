function [kMAP,logEvidences] = compMAPweights_ASDmodel(x,y,dims,minlens,nxcirc,hprs)
% Compute MAP estimate of ASD model weights given setting of hyperparameters
%
% kMAP = compMAPweights_ASDmodel(x,y,dims,minlens,nxcirc,hprs)
% 
% INPUT:
% -----
%        x [nt x nx] - stimulus matrix; each row contains is single vector of regressors
%         y [nt x 1] - response vector
%       dims [n x 1] - dims for stimulus, specifying how to reshape each x row to an image
%    minlens [1 x 1] or [n x 1] - minimum length scale for each dimension (can be scalar)
%     nxcirc [2 x 1] - circular boundary for each dimension (OPTIONAL: set to [] to use default)
%    hprs [nv x 3] - [lengthscale, rho, nsevar] (vector of hyperparameters)

% OUTPUT:
% ------
%          kMAP [n x nv] - MAP weight estimates at each setting of hyperparams
%  logEvidences [nv x 1] - log-evidence at each setting

CONDTHRESH = 1e8;  % threshold on condition number for pruning small eigenvalues of X^T X

% Parse inputs 
if isempty(nxcirc)
    nxcirc = ceil(max([dims(:)'+minlens(:)'*4;dims(:)'*1.25]))';
end
nkd = length(dims); % number of filter dimension
nktot = prod(dims);

% compute sufficient statistics, projected into Fourier domain
[dd,wwnrm,Bfft,Bii] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,CONDTHRESH);
wwnrmtot = sum(wwnrm,2); % vector of squared Fourier frequencies


%% ========= allocate space =====================

% Flip to row vector if hprs passed as column vector
if size(hprs,2) == 1
    hprs = hprs';
end
nv = size(hprs,1); % number of different hyperparameter settings

% Allocate space for log evidence
kMAP = zeros(nktot,nv);
logEvidences = zeros(nv,1);

%% ========== Evaluate log-evidence on 3D grid =============

% Make handle for loss function
lfun = @(prs)neglogev_ASDspectral(prs,dd,wwnrmtot,CONDTHRESH);  % loss function

for jv = 1:nv  % loop over hyperparameter settings
    
    % Transform rho parameter to reparametrized version
    len = hprs(jv,1); % length scale
    trho = hprs(jv,2)*(len*sqrt(2*pi)).^(nkd); % transformed rho parameter ("tilde rho")
    tildehprs = [len, trho, hprs(jv,3)]; % hyperparmeters (transformed)

    % Compute evidence and MAP weights 
    [neglogEv,~,~,muFFT,~,ii] = lfun(tildehprs);
    logEvidences(jv)= -neglogEv;

    % Transform MAP estimate from Fourier domain back to space-time domain
    Binds = false(size(Bii)); % initialize
    Binds(Bii) = ii; % indices in basis
    kest = kronmult(Bfft,muFFT,Binds); % project to space-time basis by inverse DFT
    
    % Store it
    kMAP(:,jv) = kest(:);

end