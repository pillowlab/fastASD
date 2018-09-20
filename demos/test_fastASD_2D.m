%% test_fastASD_2D.m
%
% Illustrate ASD for a 2D matrix of regression weights

% NOTE: Run script 'setpaths.m' in parent directory before running

% Generate true filter vector k
nks = [25 20];  % number of filter pixels along [cols, rows]
nk = prod(nks); % total number of filter coeffs
len = [3 3];  % length scale along each dimension
rho = 25;  % marginal prior variance

% Generate factored ASD prior covariance matrix in Fourier domain
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nks(1)); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nks(2)); % rows
nf1 = length(cdiag1); % number of frequencies needed
nf2 = length(cdiag2); 

% Draw true regression coeffs 'k' by sampling from ASD prior 
kFourier = sqrt(rho)*randn(nf1,nf2).*(sqrt(cdiag1)*sqrt(cdiag2)'); % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',nk,nf1*nf2);

% Inverse Fourier transform
kimage = U1*(U2*kFourier')'; % convert to space domain (as 2D image )
ktrue = kimage(:);  % true filter (as vector)

% Make full covariance matrix (for inspection purposes only; will cause
% out-of-memory error if filter dimensions too big!)
C1 = U1*diag(cdiag1)*U1';
C2 = U2*diag(cdiag2)*U2';
Cprior = rho*kron(C2,C1);


%%  Make stimulus and response
nsamps = 500; % number of stimulus sample
signse = 10;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*ktrue + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(221);  % ------
imagesc(Cprior);  title('prior covariance');
subplot(223); % ------
imagesc(kimage); xlabel('index'); ylabel('filter coeff'); title('true filter');
subplot(224); % ------
plot(x*ktrue, x*ktrue, 'k.', x*ktrue, y, 'r.'); xlabel('noiseless y'); ylabel('observed y');

%% Compute ridge regression estimate 
fprintf('\n...Running ridge regression with fixed-point updates...\n');

% Sufficient statistics (old way of doing it, not used for ASD)
dd.xx = x'*x;   % stimulus auto-covariance
dd.xy = (x'*y); % stimulus-response cross-covariance
dd.yy = y'*y;   % marginal response variance
dd.nx = nk;     % number of dimensions in stimulus
dd.ny = nsamps;  % total number of samples

% Run ridge regression using fixed-point update of hyperparameters
maxiter = 100;
tic;
kridge = autoRidgeRegress_fp(dd,maxiter);
toc;


%% Compute ASD estimate
fprintf('\n\n...Running ASD_2D...\n');

minlens = [2;2];  % minimum length scale along each dimension

tic; 
[kasd,asdstats] = fastASD(x,y,nks,minlens);
toc;

%%  ---- Make Plots ----

subplot(222);
imagesc(reshape(kridge,nks))
title('ridge');

subplot(224);
imagesc(reshape(kasd,nks))
title('ASD');

% Display facts about estimate
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n-----------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f)\n',len(1),asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)\n',rho(1),asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors
err = @(khat)(sum((ktrue-khat(:)).^2)); % Define error function
fprintf('\nErrors:\n------\n  Ridge = %7.2f\n  ASD2D = %7.2f\n\n', ...
     [err(kridge) err(kasd)]);
% 
