function [wwnrm,inds,wwvecs,wwinds,Bfft] = compASDfreqs(ksz,len,nxcirc,condthresh)
% Compute frequencies and diagonal elements for each of several ASD covariances
%
% [cdiags,wvecs] = mkcov_ASDfactored(prs,nx,opts)
%
% Covariance represented as C = U*sdiag*U'
% where U is unitary (in some larger basis) and sdiag is diagonal
%
%  C_ij = rho*exp(((i-j)^2/(2*l^2))
%
% INPUT:
% ------
%        ksz [m x 1] - filter dimensions
%        len [m x 1] - ASD length scale(s)
%     nxcirc [m x 1] - circular boundar
% condthresh [1 x 1] - threshold for condition number of prior covariance
%
% OUTPUT:
% -------
%   wwnrm - matrix with normalized Fourier frequencies squared, for each dimension
%    inds - binary vector with indices from the full kronecker prior covariance
%  wwvecs - cell array of normalized Fourier freqs for each dimension
%  wwinds - cell array of indices kept for each dimension in wwvecs

nd = length(ksz);  % number of dimensions

% compute vector of Fourier frequencies
maxfreq = floor(nxcirc./(pi*len)*sqrt(.5*log(condthresh))); % max frequency to keep

wwvecs = cell(nd,1); % vector of Fourier freqs for each dimension
cdiag = cell(nd,1); % diagonal of ASD prior for each dimension
wwinds = cell(nd,1); % indices to keep (of full DFT) for each dimension
Bfft = cell(nd,1); % Fourier basis matrix for each filter dimension
ncoeff = zeros(nd,1); % number of coefficients per dimension

% compute vector of Fourier frequencies
for jj = 1:nd
    if maxfreq(jj) < (nxcirc(jj)-1)/2
        ncos = maxfreq(jj); % number of cosine terms
        nsin = maxfreq(jj); % number of sine terms
    else
        % in case cutoff is above max number of frequencies
        ncos = ceil((nxcirc-1)/2); % # neg frequenceis
        nsin = floor((nxcirc-1)/2); % # pos frequencies
    end
    wwvecs{jj} = (([0:ncos, -nsin:-1])'*(2*pi/nxcirc(jj))).^2; % normalized freq
    cdiag{jj} = exp(-.5*wwvecs{jj}*len(jj)^2); % ASD diagonal (normalized)

    % vector of frequencies
    iifreq = [1:ncos+1,nxcirc(jj)-nsin+1:nxcirc(jj)]; % indices of freqs
    wwinds{jj} = iifreq;
    
    % Make realfftbasis if necessary
    if nargout >=5 
        Bfft{jj} = realfftbasis(ksz(jj),nxcirc(jj),wwvecs{jj});
    end
    
end

Cdiagfull = multikron(cdiag);
inds = Cdiagfull>1/condthresh; % indices to keep

switch nd
    case 1
        wwnrm = wwvecs{1};
    case 2
        [ww1,ww2] = ndgrid(wwvecs{1},wwvecs{2});
        wwnrm = [ww1(inds),ww2(inds)];
    case 3
        [ww1,ww2,ww3] = ndgrid(wwvecs{1},wwvecs{2},wwvecs{3});
        wwnrm = [ww1(inds),ww2(inds),ww3(inds)];
    case 4
        [ww1,ww2,ww3,ww4] = ndgrid(wwvecs{1},wwvecs{2},wwvecs{3},wwvecs{4});
        wwnrm = [ww1(inds),ww2(inds),ww3(inds),ww4(inds)];
    case 5
        error('No implementation yet for 5 or more dimensions');
end

