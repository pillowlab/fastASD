%% test_fastASD_2Dnonunif.m
%
% Test automatic smoothness determination for a 2D matrix of regression
% weights with non-uniform (irregular) spacing.

% NOTE: Run script 'setpaths.m' in parent directory before running

% Generate true filter vector k
nkgs = [50 30];
nkgrid = prod(nkgs);
rho = 4;
len = [8 8];

C1 = mkcov_ASD(len(1),1,nkgs(1)); % prior covariance matrix 
C2 = mkcov_ASD(len(2),rho,nkgs(2)); % prior covariance matrix 
Cprior = kron(C2,C1);
kgrid = mvnrnd(zeros(1,nkgrid),Cprior)'; % sample k from mvnormal with this covariance
kim = reshape(kgrid,nkgs);

%% Create non-uniform samples
nk = 603; % number of weights in nk
xpos = [sort(rand(nk,1)*nkgs(1)), rand(nk,1)*nkgs(2)]; % real-valued location for each weight / stimulus coefficient
k = interp2((0:(nkgs(2)-1)), (0:(nkgs(1)-1))',kim,xpos(:,2),xpos(:,1),'spline'); % interpolated value of k at these points

% Just to compare that things look kosher, let's visualize
xposint = ceil(xpos);
ksamp = sparse(xposint(:,1),xposint(:,2),k,nkgs(1),nkgs(2));

%%  Make stimulus and response
nsamps = 500; % number of stimulus sample
signse = 3;   % stdev of added noise
x = randn(nsamps,nk);  % stimulus
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nkgrid;
subplot(221); % ------
imagesc(kim); xlabel('index'); ylabel('filter coeff'); title('gridded filter');
subplot(222); % ------
imagesc(ksamp); xlabel('index'); ylabel('filter coeff'); title('sampled filter');
subplot(223);  % ------
plot(1:nkgs(1),kim,'k', xpos(:,1), k,'o');
title('vertical slices');
subplot(224); % ------
plot(x*k, x*k, 'k.', x*k, y, 'r.'); xlabel('noiseless y'); ylabel('observed y');

%% Compute ridge regression estimate 
fprintf('\n...Running ridge regression with fixed-point updates...\n');

% Sufficient statistics (old way of doing it, not used for ASD)
dd.xx = x'*x;   % stimulus auto-covariance
dd.xy = (x'*y); % stimulus-response cross-covariance
dd.yy = y'*y;   % marginal response variance
dd.nx = nkgrid;     % number of dimensions in stimulus
dd.ny = nsamps;  % total number of samples

% Run ridge regression using fixed-point update of hyperparameters
maxiter = 100;
tic;
kridge = autoRidgeRegress_fp(dd,maxiter);
toc;


%% Compute ASD estimate
fprintf('\n\n...Running ASD_2Dnu...\n');

minlens = 5;  % minimum length scale for each dimension
tic; 
[kasd,asdstats] = fastASD_nu(x,y,xpos,minlens);
toc;

%%  ---- Make Plots ----

% subplot(223);
% kridgesamp = sparse(xposint(:,1),xposint(:,2),kridge,nkgs(1),nkgs(2));
% imagesc(kridgesamp);
% title('ridge');
% subplot(223);
% kasdsamp = sparse(xposint(:,1),xposint(:,2),kasd,nkgs(1),nkgs(2));
% imagesc(kasdsamp);
% title('ASD');

subplot(212);
[ksrt,iisrt] = sort(k);
h = plot(1:nk,ksrt,'k-',1:nk,kridge(iisrt),'-',1:nk,kasd(iisrt), '-');
set(h(1),'linewidth',3);
title('coeffs (sorted by rank of true filter)');
legend('true', 'ridge', 'ASD 2Dnu','location', 'northwest');
axis tight;
xlabel('coeff #');

%% 
fprintf('\nHyerparam estimates\n------------------\n');
fprintf('   rho:  %3.1f  %5.2f\n',rho,asdstats.rho);
fprintf('  len1:  %3.1f  %5.2f\n',len(1),asdstats.len(1));
fprintf('signse: %3.1f  %5.2f\n',signse,sqrt(asdstats.nsevar));
% 
% 
err = @(khat)(sum((k-khat(:)).^2)); % Define error function
fprintf('\nErrors:\n  Ridge = %7.2f\n  ASD2 = %7.2f\n', ...
     [err(kridge) err(kasd)]);

% Display facts about estimate
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n-----------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f)\n',len(1),asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)\n',rho(1),asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors
err = @(khat)(sum((k-khat(:)).^2)); % Define error function
fprintf('\nErrors:\n------\n    Ridge =%7.1f\n ASD_2Dnu =%7.1f\n\n', ...
     [err(kridge) err(kasd)]);
% 
