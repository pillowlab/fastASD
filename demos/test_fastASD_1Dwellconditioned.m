% test_fastASD_1Dwellconditioned.m
%
% Check code that computes evidence without truncating small eigenvalues
% (using a "well-conditioned" version of the ASD prior covariance matrix
% that has a small constant diagonal added to it).
%
% Uses simulated example from test_fastASD_1D.m

% NOTE: Run script 'setpaths.m' in parent directory before running


%% Generate simulated dataset

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


%% Compute ASD estimate 
fprintf('\n\n...Running ASD...\n');

% Set lower bound on length scale.  (Larger -> faster inference).
% If it's set too high, the function will warn that optimal length scale is
% at this lower bound, and you should re-run with a smaller value since
% this may be too high
minlen = 20; % minimum length scale

% Run ASD and return sufficient statistics 
[kasd,asdstats,dd] = fastASD(x,y,nk,minlen);
[kasd_wc1,asdstats_wc1,dd_wc1] = fastASD_wellcond(x,y,nk,minlen,[],1e-7);
[kasd_wc2,asdstats_wc2,dd_wc2] = fastASD_wellcond(x,y,nk,minlen,[],1e-3);


%% Plot + Report Results

subplot(221);
h = plot(1:nk, k_true, 'k', 1:nk, kasd, 1:nk, kasd_wc1, '--', 1:nk, kasd_wc2, '-.');
set(h(2:3),'linewidth', 2);
legend('true k', 'asd estimate', 'well-cond asd1', 'well-cond asd2');
title('MAP estimates');

subplot(222);
h = plot(1:nk, kasd-kasd, 1:nk, kasd-kasd_wc1, '--', 1:nk, kasd-kasd_wc2, '-.');
set(h(1:2),'linewidth', 2);
legend('asd estimate', 'well-cond asd1', 'well-cond asd2');
title('MAP estimate errors');

subplot(212);
nf = length(dd.wwnrm); % number of fourier coeffs
cdiag1 = zeros(nf,1); 
cdiag2 = zeros(nf,1);
cdiag3 = zeros(nf,1);
cdiag1(dd.ii) = 1./diag(dd.Cinv);
cdiag2(dd_wc1.ii) = 1./diag(dd_wc1.Cinv);
cdiag3(dd_wc2.ii) = 1./diag(dd_wc2.Cinv);
semilogy(1:nfourier,cdiag1, 'o-', 1:nfourier, cdiag2, '--', 1:nf,cdiag3, '-.','linewidth', 2)
axis tight;
title('Cdiag entries of learned prior covariance');
legend('asd', 'well-cond asd1', 'well-cond asd2');

fprintf('\nEvidence values:\n=================\n');
fprintf('Standard ASD: %.2f\n', -asdstats.neglogEv);
fprintf('Wellcond ASD1: %.2f\n', -asdstats_wc1.neglogEv);
fprintf('Wellcond ASD2: %.2f\n', -asdstats_wc2.neglogEv);

fprintf('\nLearned length scales:\n=================\n');
fprintf('Standard ASD: %.2f\n', -asdstats.len);
fprintf('Wellcond ASD1: %.2f\n', -asdstats_wc1.len);
fprintf('Wellcond ASD2: %.2f\n', -asdstats_wc2.len);

