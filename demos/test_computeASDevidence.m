% test_computeASDevidence.m
%
% Script illustrating evaluation of log-evidence function at different
% values of model hyperparameters. 
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
[k_asd,asdstats] = fastASD(x,y,nk,minlen);


%% Compute log-evidence on 1D grid of length scales

lenvals = [minlen:2.5:80, 85:5:150, 175 200];
rhovals = asdstats.rho;
nsevals = asdstats.nsevar;

[logEvs,khats] = compLogEvidence_ASDmodel(x,y,nk,minlen,[],lenvals,rhovals,nsevals);
subplot(221); 
plot(lenvals, logEvs,asdstats.len,-asdstats.neglogEv,'k*'); 
xlabel('length scale'); axis tight;
ylabel('log evidence');
title('log-evidence vs. length scale');

subplot(222);
h = plot(1:nk, k_asd, 'k--', 1:nk,khats);
set(h(1),'linewidth',3);
xlabel('pixel #');
ylabel('coefficient');
title('MAP estimates at different lengthscales');

%% Compute log-evidence on 2D grid of length scale & Rho

lenvals = asdstats.len + (-10:.5:10);
rhovals = asdstats.rho + (-0.4:.05:1);
nsevals = asdstats.nsevar;

logEvs2 = compLogEvidence_ASDmodel(x,y,nk,minlen,[],lenvals,rhovals,nsevals);
subplot(223);
imagesc(lenvals,rhovals,logEvs2');  axis xy;
hold on;  plot(asdstats.len,asdstats.rho,'k*'); hold off;
xlabel('length scales');
ylabel('rhos');
title('log-evidence vs. length scale and rho');

%% Compute log-evidence on 2D grid of Rho & noise variance

lenvals = asdstats.len;
rhovals = asdstats.rho + (-0.4:.05:1);
nsevals = asdstats.nsevar + (-10:.2:10);

logEvs3 = squeeze(compLogEvidence_ASDmodel(x,y,nk,minlen,[],lenvals,rhovals,nsevals));
subplot(224);
imagesc(nsevals,rhovals,logEvs3);  axis xy;
hold on;  plot(asdstats.nsevar,asdstats.rho,'k*'); hold off;
xlabel('nsevar');
ylabel('rho');
title('log-evidence vs. rho and noise variance');



