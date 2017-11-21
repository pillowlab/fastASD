%% test_fastASD_2Dnonisotropic.m
%
% Test automatic smoothness determination for a 2D matrix of regression
% weights with non-isotropic smoothness (diff length scale along each dimension)

% NOTE: Run script 'setpaths.m' in parent directory before running

% Generate true filter vector k
nks = [30 35];
nk = prod(nks);
len = [3 15];
rho = 4; 

% Generate factored ASD prior covariance matrix in Fourier domain
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nks(1)); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nks(2)); % rows
nf1 = length(cdiag1); % number of frequencies needed
nf2 = length(cdiag2); 

% Draw true regression coeffs 'k' by sampling from ASD prior 
kh = sqrt(rho)*randn(nf1,nf2).*(sqrt(cdiag1)*sqrt(cdiag2)'); % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',nk,nf1*nf2);

% Inverse Fourier transform
kim = U1*(U2*kh')'; % convert to space domain (as 2D image )
k = kim(:);  % as vector

%%  Make stimulus and response
nsamps = 500; % number of stimulus sample
signse = 10;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(221); % ------
imagesc(kim); xlabel('index'); ylabel('filter coeff'); title('true filter');
subplot(222); % ------
plot(x*k, x*k, 'k.', x*k, y, 'r.'); xlabel('noiseless y'); ylabel('observed y');

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


%% Compute isotropic ASD estimate
fprintf('\n\n...Running isotropic ASD_2D...\n');

minlens = [2];  % minimum length scale along each dimension
tic; 
[kasd,asdstats] = fastASD(x,y,nks,minlens);
toc;
%% Compute full ASD estimate
fprintf('\n\n...Running full ASD_2D...\n');

minlens2 = [2;10];  % minimum length scale along each dimension
tic; 
[kasd2,asdstats2] = fastASD_aniso(x,y,nks,minlens2);
toc;


%%  ---- Make Plots ----

subplot(222);
imagesc(reshape(kridge,nks))
title('ridge');

subplot(223);
imagesc(reshape(kasd,nks))
title('isotropic ASD');

subplot(224);
imagesc(reshape(kasd2,nks))
title('full ASD');

% Display facts about estimate
ci = asdstats.ci;
ci2 = asdstats2.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n-----------------------\n');
fprintf('        True    isotropic         full  \n');
fprintf('    l1: %5.1f  %5.1f (+/-%.1f)',len(1),asdstats.len,ci(1));
fprintf(' %5.1f (+/-%.1f)\n',asdstats2.len(1),ci2(1));
fprintf('    l2: %5.1f    ---         ',len(2)); 
fprintf(' %5.1f (+/-%.1f)\n',asdstats2.len(2),ci2(2));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)',rho,asdstats.rho,ci(2));
fprintf(' %5.1f (+/-%.1f)\n',asdstats2.rho,ci2(3));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)',signse.^2,asdstats.nsevar,ci(3));
fprintf(' %5.1f (+/-%.1f)\n',asdstats2.nsevar,ci2(4));

% Compute errors
%err = @(khat)(1-(norm(k-khat(:))/norm(k))^2); % Define error function
err = @(khat)(sum((k-khat(:)).^2)); % Define error function
fprintf(['\nErrors:\n------\n', ...
         '  Ridge   = %4.3f\n' ...
         '  ASDiso  = %4.3f\n' ...
         '  ASDfull = %4.3f\n\n'], [err(kridge) err(kasd) err(kasd2)]);
% 
