%% test_fastASD_1Dnonunif.m
%
% Illustrate automatic smoothness determination (ASD) on 1D set of
% weights with non-uniform spacing

% NOTE: Run script 'setpaths.m' in parent directory before running

% Generate true filter vector k on a lattice
nkgrid = 500;  % number of filter coeffs for full filter
rho = 2; % marginal variance
len = 20;  % ASD length scale

C0 = mkcov_ASD(len,rho,nkgrid); % prior covariance matrix 
kgrid = mvnrnd(zeros(1,nkgrid),C0)'; % sample k from mvnormal with this covariance

% Create non-uniform samples
nk = 403; % number of weights in nk
xpos = sort(rand(nk,1)*nkgrid); % real-valued location for each weight / stimulus coefficient
k = interp1(0:(nkgrid-1),kgrid,xpos,'spline'); % interpolated value of k at these points

% % Create non-uniform samples
% nk = nkgrid; % number of weights in nk
% xpos = (0:(nk-1))'; % real-valued location for each weight / stimulus coefficient
% k = kgrid;


%%  Make stimulus and simluate responses
nsamps = 500; % number of stimulus sample
signse = 10;   % stdev of added noise
x = randn(nsamps,nk); % stimulus
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
xgrid = 0:(nkgrid-1);
subplot(211); plot(xgrid,kgrid,xpos,k,'*'); % show original and unevenly-sampled k

xlabel('index'); ylabel('filter coeff');
title('true filter');
legend('on grid', 'non-uniform samples');

subplot(212); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');

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
kridge = autoRidgeRegress_fp(dd,100);
toc;

%% Compute ASD estimate 
fprintf('\n\n...Running ASD...\n');

% Set lower bound on length scale, which is distance at which the
% correlation will have falledn by exp(-1).
% Should be in same units as xi.
% Larger => Faster.  (Code will warn if optimal value needs to be lower).
minlen = 10; 

% Run ASD
tic;
[kasd,asdstats] = fastASD_nu(x,y,xpos,minlen);
toc;

% % Pedagogical note: using a*xpos and a*minlen, for any a>0, gives identical estimate
% [kasd2,asdstats2] = fastASD_1Dnu(x,y,xpos*2,minlen*2);
% plot([kasd, kasd2]);

%%  ---- Make Plots ----

h = plot(xpos,k,'k-',xpos,kridge,xpos,kasd,'r');
set(h(1),'linewidth',2.5);
title('estimates');
legend('true', 'ridge', 'ASD');

% Display facts about estimate
ci = asdstats.ci;
set(h(1),'linewidth',2.5);
title('estimates');
legend('true', 'ridge', 'ASD');
fprintf('\nHyerparam estimates (+/-1SD)\n----------------------------\n');
fprintf('     l: %5.1f  %5.1f +/- %.1f\n',len,asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f +/- %.1f\n',rho,asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f +/- %.1f\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors 
err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n------\n  Ridge = %7.2f\n  ASD = %9.2f\n\n', [err(kridge) err(kasd)]);
