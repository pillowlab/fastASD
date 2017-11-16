function dd = compSuffStats_ASD(x,y,ksz,minlens,nxcirc,condthresh)
% Compute sufficient statistics for ASD filter estimation
%
% dd = compLSsuffstats_fourierELLdiag(x,y,dims,minlens,nxcirc)
%
% INPUT:
% -----
%           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
%           y [n x 1] - response vector
%         ksz [m x 1] - filter size (eg, [nkt x nkx x nky] for 3D spatiotemporal filter)
%     minlens [m x 1] - minimum length scale for each dimension
%      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
%  condthresh [1 x 1] - threshold on condition number (for thresholding eigenvalues)
%
% OUTPUT:
% ------
%     dd (struct) - carries sufficient statistics for linear regresion
%  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
%   Bfft  {1 x p} - cell array with DFT bases for each dimension

% Set 'condthresh' if necessary
if nargin < 6
    condthresh = 1e8; % default value (condition number on prior covariance)
end

% Set 'nxcirc' if necessary
if (nargin < 5)
    nxcirc = ceil(max([ksz(:)'+minlens(:)'*4;ksz(:)'*1.25]))';
end

% Check dimensions of filter and stimulus
xwid = size(x,2);
if prod(ksz)==xwid
    % no temporal dimension to filter
    PURESPATIAL=true;
    if (ksz(1) == 1),  % Remove first filter dimension if it's 1
        ksz = ksz(2:end); 
        minlens = minlens(2:end);
        nxcirc = nxcirc(2:end);
    end
elseif prod(ksz(2:end))==xwid
    PURESPATIAL=false; % is a spatiotemporal filter
else
    error('mismatch between size of stimulus ''x'' and filter dimensions ''dims''');
end

if PURESPATIAL

    % Determine number of freqs and make Fourier basis for each dimension
    [wwnrm,inds,wwvecs,wwinds,Bfft] = compASDfreqs(ksz,minlens,nxcirc,condthresh);
    fprintf('\n Total # Fourier coeffs represented: %d\n\n', size(wwnrm,1));
    nsamps = size(y,1);
        
    % Calculate stimulus sufficient stats in Fourier domain
    Bx = kronmult(Bfft,x'); % convert to Fourier domain
    Bx = Bx(inds,:); % prune unneeded freqs
    dd.xx = Bx*Bx';
    dd.xy = Bx*y;
    dd.yy = y'*y; % marginal response variance
    dd.nsamps = nsamps; % total number of samples
    
    % Other stuff
    dd.ksz = ksz;
    dd.wwnrm = wwnrm;
    dd.inds = inds;
    dd.Bfft = Bfft;
    dd.wwvecs = wwvecs;
    dd.wwinds = wwinds;
    dd.fftdims = cellfun(@(x)size(x,1),wwvecs);
else
  % Do temporal     
    
end


% % Fill in other statistics
% dd.yy = y'*y; % marginal response variance
% %dd.nx = xwid; % number of dimensions in stimulus
% dd.nsamps = size(y,1);  % total number of samples
% 
% 
% dd.xx = XXfftdiag;
% 
% % Compute projected mean X'*Y
% xyfft = realfft(x'*y,nxcirc);
% dd.xy = xyfft(iiw);

%  Fill in other statistics
dd.condthresh = condthresh;
dd.nxcirc = nxcirc;
dd.nxperdim = ksz(:);
dd.minlens = minlens(:);

