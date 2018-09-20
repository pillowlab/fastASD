% test_computeASDevidence_2D.m
%
% Script illustrating evaluation of log-evidence function in Fourier domain
% vs. in original domain
%
% Uses simulated example from test_fastASD_2D.m

% NOTE: Run script 'setpaths.m' in parent directory before running


%% Generate simulated dataset

% Generate true filter vector k
nks = [25 20];  % number of [rows, columns] in filter
ncoeff = prod(nks); % total number of filter coeffs
len = [3 3];  % length scale along each dimension
rho = 25;  % marginal prior variance

% Set lower bound on length scale.  (Larger -> faster inference).
minlen = 1; % minimum length scale (for inference purposes)

% Generate factored ASD prior covariance matrix in Fourier domain
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nks(1)); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nks(2)); % rows
nf1 = length(cdiag1); % number of frequencies needed
nf2 = length(cdiag2); 

% Draw true regression coeffs 'k' by sampling from ASD prior 
kh = sqrt(rho)*randn(nf1,nf2).*(sqrt(cdiag1)*sqrt(cdiag2)'); % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',ncoeff,nf1*nf2);

% Inverse Fourier transform
kimage = U1*(U2*kh')'; % convert to space domain (as 2D image )
ktrue = kimage(:);  % as vector

% % Make full covariance matrix (for inspection purposes only; will cause
% % out-of-memory error if filter dimensions too big!)
C1 = U1*diag(cdiag1)*U1'; % column covariance 
C2 = U2*diag(cdiag2)*U2'; % row covariance
Cprior = rho*kron(C2,C1); % full prior covariance

%%  Make stimulus and response
nsamps = 800; % number of stimulus sample
signse = 10;   % stdev of added noise
x = gsmooth(randn(ncoeff,nsamps),1)'; % stimulus (smooth)
y = x*ktrue + randn(nsamps,1)*signse;  % dependent variable 

%% Compute ASD estimate 
fprintf('\n\n...Running ASD...\n');
[k_asd,asdstats] = fastASD(x,y,nks,minlen);

fprintf('\n Inferred_Lengthscale = %.2f\n', asdstats.len);

%% Compute log-evidence on 1D grid of length scales

lenvals = minlen:.1:5;  % grid of lengthscale values
rhovals = rho; % use true value of rho
nsevals = signse^2; % use true value of noise variance

% Compute Fourier-domain evidence on a grid
nxcirc = nks+lenvals(end)*3; % set maximum circular boundary
[logEvs_fourier,khats] = compLogEvidence_ASDmodel(x,y,nks,minlen,nxcirc,lenvals,rhovals,nsevals);

% Use for loop to evaluate log-evidence with dual form formula on a grid

nlens = length(lenvals); 
logEvs_real = zeros(nlens,1);
Cnoise = signse^2*eye(nsamps); % covariance of added noise

for jj = 1:nlens
    
    % Compute prior covariance
    Cr = mkcov_ASD(lenvals(jj),rho,nks(1));
    Cc = mkcov_ASD(lenvals(jj),1,nks(2));
    Cp = kron(Cc,Cr);

    L = x*Cp*x' + Cnoise; % covariance
    logEvs_real(jj) = logmvnpdf(y',zeros(1,nsamps),L);

end

%%
clf;
plot(lenvals, logEvs_real, lenvals, logEvs_fourier, '--', 'linewidth', 2);
xlabel('length scale');
ylabel('log-evidence');
legend('real domain', 'Fourier domain');
box off;
title('log-evidence calculation');

