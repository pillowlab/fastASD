function [dd,wwnrm,Bfft] = compLSsuffstats_fourier_nu(x,y,xpos,minlens,nxcirc,condthresh)
% Compute LS regression sufficient statistics in DFT basis for non-uniform data
%
% [dd,wvecs,Bfft,wwnrm,B,ii] = compLSsuffstats_fourier_nu(x,y,xpos,minlens,nxcirc,condthresh)
%
% INPUT:
% -----
%           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
%           y [n x 1] - response vector
%        xloc [p x m] - spatial location of all p coeffs along each dimension m
%     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
%      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
%  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
%
% OUTPUT:
% ------
%     dd - data structure with sufficient statistics for linear regresion
%  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
%   Bfft [nf x p] - basis for 2D DFT 

% Determine size of stimulus and its dimensions
[nsamps,nx] = size(x);
nd = size(xpos,2); % number of dimensions
if length(minlens) == 1 % make vector out of minlens, if necessary
    minlens = repmat(minlens,nd,1);
end

xwid = range(xpos)'*((nsamps+1)/nsamps); % estimated support along each dimension
% Set nxcirc to default value if necessary
if (nargin < 5) || isempty(nxcirc)
    nxcirc = ceil(max([xwid(:)'+minlens(:)'*4; ...
                       xwid(:)'*1.25]))';
end

% Make Fourier basis for each input dimension 
Bmats = cell(nd,1); % Fourier basis matrix for each filter dimension
wwnrmvecs = cell(nd,1); % Fourier frequencies for each filter dimension
cdiagvecs = cell(nd,1); % eigenvalues for each dimension
for jj = 1:nd
    % Move to range [0 xwid(jj)].
    if min(xpos(:,jj))>0
        xpos(:,jj) = xpos(:,jj)-min(xpos(:,jj));
    end
    
    % determine maximal freq for Fourier representation
    maxfreq = floor(nxcirc(jj)/(pi*minlens(jj))*sqrt(.5*log(condthresh)));

    % Compute basis for non-uniform DFT and frequency vector
    [Bmats{jj},wvecs] = realnufftbasis(xpos(:,jj),nxcirc(jj),maxfreq*2+1);
    wwnrmvecs{jj} = (2*pi/nxcirc(jj))^2*(wvecs.^2); % normalized freqs squared
    cdiagvecs{jj} = exp(-.5*wwnrmvecs{jj}*minlens(jj).^2); % diagonal of cov

end


switch nd   
    case 1,  % 1 stimulus dimension

        % Convert cell to array
        Bfft = Bmats{1};      % FFT matrix
        wwnrm = wwnrmvecs{1}; % normalized freqs squared

    case 2,   % 1 stimulus dimension
        nfreq = cellfun(@length,wwnrmvecs); % number of frequencies preserved for each dimension
        Cdiag = kron(cdiagvecs{2},cdiagvecs{1}); % diagonal for full space
        ii = Cdiag>1/condthresh; % indices to keep
        [i1,i2] = find(reshape(ii,nfreq')); % find indices for wwnrm1 and wwnrm2
        % Compute outer product of basis vecs to get basis for 2D NUDFT
        Bfft = Bmats{1}(i1,:).*Bmats{2}(i2,:);
        wwnrm = [wwnrmvecs{1}(i1), wwnrmvecs{2}(i2)];
        
end

% Calculate stimulus sufficient stats in Fourier domain
nf = size(Bfft,1); % number of frequencies
if (nsamps*nf*(nx+nf)) > (nx^2*(nsamps+nf)+nf^2*nx)
    % Compute X'X and then transform
    dd.xx = Bfft*(x'*x)*Bfft';  % stimulus covariance
    dd.xy = Bfft*(x'*y);  % stimulus-response cross-covariance
else
    % Transform and then compute X'X.
    xB = x*Bfft';
    dd.xx = xB'*xB;
    dd.xy = xB'*y;
end

% Fill in other statistics
dd.yy = y'*y; % marginal response variance
dd.nx = nx; % number of dimensions in stimulus
dd.nsamps = nsamps;  % total number of samples

