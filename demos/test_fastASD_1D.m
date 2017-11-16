% test_fastASD_1D.m
%
% Illustrates fast automatic smoothness determination (ASD) for a 
% 1-dimensional (vector) receptive field (RF) sampled from the ASD prior

% Run script 'setpaths.m' in parent directory before running

% Set prior distribution for filter vector
nk = 1000; % number of filter coeffs (1D vector)
rho = 2;   % marginal variance of prior
len = 50; % length scale of prior
C0 = mkcov_ASD(len,rho,nk); % prior covariance matrix 

% Sample true filter from the prior distribution
k_true = mvnrnd(zeros(1,nk),C0)'; % sample ktrue from multivariate normal 

% Generate stimuli and sample responses from model
nsamps = 500; % number of stimulus samples
signse = 10;  % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % make smooth stimuli 
y = x*k_true + randn(nsamps,1)*signse; % simulate neural response

% plot filter and examine noise level
t = 1:nk;
subplot(211); plot(t,k_true);
xlabel('index'); ylabel('filter coeff');
title('true filter');

subplot(212); plot(x*k_true, x*k_true, 'k.', x*k_true, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');

%% Compute ridge regression estimate 
fprintf('\n...Running ridge regression with fixed-point updates...\n');

% Compute sufficient statistics 
% (needed for automatic ridge regression code; not used in fastASD)
dd.xx = x'*x;   % stimulus auto-covariance
dd.xy = (x'*y); % stimulus-response cross-covariance
dd.yy = y'*y;   % marginal response variance
dd.nx = nk;     % number of dimensions in stimulus
dd.ny = nsamps;  % total number of samples

% Run ridge regression using fixed-point update of hyperparameters
maxiter = 100;
[k_ridge,hyperparams_ridge] = autoRidgeRegress_fp(dd,maxiter);

%% Compute ASD estimate 
fprintf('\n\n...Running ASD...\n');

% Set lower bound on length scale.  (Larger -> faster inference).
% If it's set too high, the function will warn that optimal length scale is
% at this lower bound, and you should re-run with a smaller value since
% this may be too high
minlen = 25; % minimum length scale

% Run ASD
tic;
[k_asd,asdstats] = fastASD(x,y,nk,minlen);
toc;

%%  ---- Make Plots ----

subplot(211); % Plot ridge estimate
h = plot(t,k_true,'k-',t,k_ridge);
set(h(1),'linewidth',2); 
title('automatic ridge regression');
legend('true k', 'ridge estimate');
ylabel('RF coefficient');
xlabel('pixel location');

subplot(212); % Plot ASD estimate (with error bars)
h = plot(t,k_true,'k-',t,k_asd,'r','linewidth', 2); 
set(h(1),'linewidth', 2); hold on;
kasdSD = sqrt(asdstats.Lpostdiag); % posterior stdev for asd estimate
errorbarFill(t,k_asd,2*kasdSD); % plot posterior marginal confidence intervals
hold off;
legend('true k','ASD estimate')
title('ASD (+/-2SD error bars)');
xlabel('pixel location');
ylabel('RF coefficient');

% Examine hyperparameter recovery
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n----------------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f)\n',len,asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)\n',rho,asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors 
l2err = @(k1,k2)(sum((k1(:)-k2(:)).^2)); % Define sum of squared errors function
r2fun = @(k)(1-l2err(k_true,k)/l2err(k_true,mean(k_true(:)))); % R^2 function
fprintf('\nR-squared:\n----------\n');
fprintf('  Ridge = %5.2f\n    ASD = %5.2f\n\n', [r2fun(k_ridge) r2fun(k_asd)]);
