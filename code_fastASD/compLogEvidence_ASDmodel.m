function [logEvidences,khats] = compLogEvidence_ASDmodel(x,y,dims,minlens,nxcirc,lenvals,rhovals,nsevarvals)
% Evaluate ASD log-evidence on a grid of hyperparameter values
%
% logEvidences = compLogEvidence_ASDmodel(x,y,dims,minlens,nxcirc,lenvals,rhovals,nsevarvals)
% 
% INPUT:
% -----
%        x [nt x nx] - stimulus matrix; each row contains is single vector of regressors
%         y [nt x 1] - response vector
%       dims [n x 1] - dims for stimulus, specifying how to reshape each x row to an image
%    minlens [1 x 1] or [n x 1] - minimum length scale for each dimension (can be scalar)
%     nxcirc [2 x 1] - circular boundary for each dimension (OPTIONAL: set to [] to use default)
%    lenvals [nl x 1] - vector of length-scales
%    rhovals [nr x 1] - vector of rhos
% nsevarvals [nn x 1] - vector of noise variances (sig^2)
%
% OUTPUT:
% ------
%  logEvidences [nr x nl x nn] - tensor of log-evidences
%         khats [n x nl x nr x nn] - MAP weight estimates at gridded hyperparam values

CONDTHRESH = 1e8;  % threshold on condition number for pruning small eigenvalues of X^T X

% Parse inputs 
if isempty(nxcirc)
    nxcirc = ceil(max([dims(:)'+minlens(:)'*6;dims(:)'*1.33]))';
end
nkd = length(dims); % number of filter dimension
nktot = prod(dims);

% compute sufficient statistics, projected into Fourier domain
[dd,wwnrm,Bfft,Bii] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,CONDTHRESH);
wwnrmtot = sum(wwnrm,2); % vector of squared Fourier frequencies


%% ========= allocate space =====================
nl = length(lenvals); % number of lengthscales
nr = length(rhovals); % number of rhos
nn = length(nsevarvals); % number of noisevars

% Allocate space for log evidence
logEvidences = zeros(nl,nr,nn); 

% Allocate space for RF estimates at each setting
if nargout > 1
    khats = zeros(nktot,nl,nr,nn);
end

%% ========== Evaluate log-evidence on 3D grid =============

% Make handle for loss function
lfun = @(prs)neglogev_ASDspectral(prs,dd,wwnrmtot,CONDTHRESH);  % loss function

for jl = 1:nl  % loop over length scale
    len = lenvals(jl);

    for jr = 1:nr  % loop over rho 
        % Change of variables from "rho" to "tilde rho", which separates rho and length scale
        trho = rhovals(jr)*(len*sqrt(2*pi)).^(nkd);
        
        for jn = 1:nn % loop over noise values
            nsevar = nsevarvals(jn);
            
            % Set hyperparameters
            hprs = [len,trho,nsevar];

            if nargout == 1
                % Compute evidence only
                logEvidences(jl,jr,jn) = - lfun(hprs);

            else
                % Compute evidence and MAP weights (in Fourier domain)
                [neglogEv,~,~,muFFT,~,ii] = lfun(hprs);
                logEvidences(jl,jr,jn) = -neglogEv;                

                % Transform MAP estimate back to space-time domain
                Binds = false(size(Bii)); % initialize
                Binds(Bii) = ii; % indices in basis
                kest = kronmult(Bfft,muFFT,Binds); % project to space-time basis by inverse DFT

                % Store it
                khats(:,jl,jr,jn) = kest;
            end
        end
    end
end
