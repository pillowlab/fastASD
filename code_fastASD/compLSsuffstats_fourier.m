function [dd,wwnrm,Bfft,ii] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,condthresh)
% Compute least-squares regression sufficient statistics in DFT basis
%
% [dd,wwnrm,Bfft] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,condthresh)
%
% INPUT:
% -----
%           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
%           y [n x 1] - response vector
%        dims [m x 1] - number of coefficients along each stimulus dimension
%     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
%      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
%  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
%
% OUTPUT:
% ------
%     dd (struct) - carries sufficient statistics for linear regresion
%  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
%   Bfft  {1 x p} - cell array with DFT bases for each dimension


% Check if optional inputs passed in
if nargin < 6
    condthresh = 1e8; % default value (condition number on prior covariance)
end

% Set circular bounardy (for n-point fft) to avoid edge effects, if needed
if (nargin < 5) || isempty(nxcirc)
    nxcirc = ceil(max([dims(:)'+minlens(:)'*4; ...
                       dims(:)'*1.25]))';
end


nd = length(dims); % number of filter dimensions
if length(minlens) == 1 % make vector out of minlens, if necessary
    minlens = repmat(minlens,nd,1);
end

% Determine number of freqs and make Fourier basis for each dimension
cdiagvecs = cell(nd,1); % eigenvalues for each dimension
Bfft = cell(nd,1); % Fourier basis matrix for each filter dimension
wvecs = cell(nd,1); % Fourier frequencies for each filter dimension
ncoeff = zeros(nd,1);
opt1.condthresh = condthresh;
fprintf('\ncompLSsuffstats_fourier:\n # filter freqs per stimulus dim:');
% Loop through dimensions
for jj = 1:nd
    opt1.nxcirc = nxcirc(jj); 
    [cdiagvecs{jj},Bfft{jj},wvecs{jj}] = mkcov_ASDfactored([minlens(jj);1],dims(jj),opt1);
    ncoeff(jj) = length(cdiagvecs{jj}); % number of coeffs
    fprintf(' %d ', ncoeff(jj));
end
fprintf('\n Total # Fourier coeffs represented: %d\n\n', prod(ncoeff));

switch nd  
    % switch based on stimulus dimension

    case 1, % 1 dimensional stimulus
        wwnrm = (2*pi/nxcirc(1))^2*(wvecs{1}.^2); % normalized freqs squared
        ii = true(length(wwnrm),1)'; % indices to keep 
        
    case 2, % 2 dimensional stimulus
        
        % Form full frequency vector and see which to cut
        Cdiag = kron(cdiagvecs{2},cdiagvecs{1});
        ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep 
                    
        % compute vector of normalized frequencies squared
        [ww1,ww2] = ndgrid(wvecs{1},wvecs{2});
        wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2 ...
            (ww2(ii)*(2*pi/nxcirc(2))).^2];
        
    case 3, % 3 dimensional stimulus

        Cdiag = kron(cdiagvecs{3},(kron(cdiagvecs{2},cdiagvecs{1})));
        ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep
        
        % compute vector of normalized frequencies squared
        [ww1,ww2,ww3] = ndgrid(wvecs{1},wvecs{2},wvecs{3});
        wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2, ...
            (ww2(ii)*(2*pi/nxcirc(2))).^2, ....,
            (ww3(ii)*(2*pi/nxcirc(3))).^2];
        
    otherwise
        error('compLSsuffstats_fourier.m : doesn''t yet handle %d dimensional filters\n',nd);
        
end

% Calculate stimulus sufficient stats in Fourier domain
Bx = kronmulttrp(Bfft,x'); % convert to Fourier domain
Bx = Bx(ii,:); % prune unneeded freqs
dd.xx = Bx*Bx';
dd.xy = Bx*y;

% Fill in other statistics
dd.yy = y'*y; % marginal response variance
%dd.nx = xwid; % number of dimensions in stimulus
dd.nsamps = size(y,1);  % total number of samples


% ------ Examine speed of knonmult for DFT operation ----------
% [nsamps,xwid] = size(x); % Determine size of stimulus and its dimensions
% % Relative cost of FFT first vs. x'*x first if we did full kronecker
% (nsamps*nf*(xwid+nf)) > (xwid^2*(nsamps+nf)+nf^2*xwid)
% % Old / slow way:  
% xx = x'*x;  xy = x'*y;  then FFT
% xx = kronmult(Bfft,kronmult(Bfft,xx)')';
% xy = kronmult(Bfft,xy);
% -------------------------------------------------------------        
