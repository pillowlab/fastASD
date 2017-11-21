%% test_fastASD_1Dgroupedcoeffs.m
%
% Illustrate fast automatic smoothness determination with regression
% weights composed of multiple (unrelated) groups of smooth coefficients

% NOTE: Run script 'setpaths.m' in parent directory before running

% Generate true filter vector k
nkgrp = [600,1000,400]; % number of coefficients in each group
ngrp = length(nkgrp);
nk = sum(nkgrp);
len = [40,120,15];  % ASD length scale
rho = [2,1,4]; % marginal variance

Cgrp = [];
Cgrp{1} = mkcov_ASD(len(1),rho(1),nkgrp(1)); % prior covariance matrix 
Cgrp{2} = mkcov_ASD(len(2),rho(2),nkgrp(2)); % prior covariance matrix 
Cgrp{3} = mkcov_ASD(len(3),rho(3),nkgrp(3)); % prior covariance matrix 
Cprior = blkdiag(Cgrp{:});
k = mvnrnd(zeros(1,nk),Cprior)'; % sample k from mvnormal with this covariance

%  Make stimulus and response
nsamps = 500; % number of stimulus sample
signse = 10;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(221);  % ------
imagesc(Cprior);  title('prior covariance');
subplot(223); % ------
plot(t,k); xlabel('index'); ylabel('filter coeff'); title('true filter');
subplot(224); % ------
plot(x*k, x*k, 'k.', x*k, y, 'r.'); xlabel('noiseless y'); ylabel('observed y');

%% Run inference

% ----- Run standard ASD ----
fprintf('\n\n...Running standard ASD...\n');
minlen = 10; 
tic;
[kasd1,asdstats1] = fastASD(x,y,nk,minlen);
toc;

% ----- Run group ASD --------
fprintf('\n\n...Running group ASD...\n');

minlen = 10;
tic; 
[kasd2,asdstats2] = fastASD_group(x,y,nkgrp,minlen);
toc;


%%  ---- Make Plots ----

clf;
% h = plot(t,k,'k-',t,kasd1,t,kasd2);
% set(h(1),'linewidth',2.5);


kasdSD = sqrt(asdstats2.Lpostdiag); % posterior stdev for asd estimate
plot(t,k,'k',t,kasd1,t,kasd2); hold on;
errorbarFill(t,kasd2,2*kasdSD); % plot posterior marginal confidence intervals
hold off;
title('estimates');
legend('true', 'ASD','groupASD');



% Display facts about estimate
ci = asdstats2.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n----------------------------\n');
fprintf('     l1:  %5.1f %5.1f (+/-%.1f)\n',len(1),asdstats2.len(1),ci(1));
fprintf('     l2:  %5.1f %5.1f (+/-%.1f)\n',len(2),asdstats2.len(2),ci(2));
fprintf('     l3:  %5.1f %5.1f (+/-%.1f)\n',len(3),asdstats2.len(3),ci(3));
fprintf('   rho1:  %5.1f %5.1f (+/-%.1f)\n',rho(1),asdstats2.rho(1),ci(4));
fprintf('   rho2:  %5.1f %5.1f (+/-%.1f)\n',rho(2),asdstats2.rho(2),ci(5));
fprintf('   rho3:  %5.1f %5.1f (+/-%.1f)\n',rho(3),asdstats2.rho(3),ci(6));
fprintf(' signse: %3.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats2.nsevar,ci(7));

% Compute errors 
err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n-------\n       ASD = %7.2f\n  groupASD = %7.2f\n', ...
    [err(kasd1) err(kasd2)]);

