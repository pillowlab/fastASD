%% test_fastASD_3D.m
%
% Test automatic smoothness determination (ASD) for a 3D tensor of regression weights

% NOTE: Run script 'setpaths.m' in parent directory before running

% Generate true filter vector k
nks = [12 13 10]; % number of filter pixels along [cols, rows]
nk = prod(nks); % total number of filter coeffs
len = 3.5*[1 1 1];   % length scale along each dimension
rho = 3;  % marginal prior variance

% Generate factored ASD prior covariance matrix in Fourier domain
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nks(1)); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nks(2)); % rows
[cdiag3,U3,wvec3] = mkcov_ASDfactored([len(3),1],nks(3)); % rows
nf1 = length(cdiag1); % number of frequencies needed
nf2 = length(cdiag2); 
nf3 = length(cdiag3); 
fftdms = [nf1 nf2 nf3];

% Draw true regression coeffs 'k' by sampling from ASD prior 
Csqrt = sqrt(rho)*reshape(kron(sqrt(cdiag3),kron(sqrt(cdiag2),sqrt(cdiag1))),fftdms);
kh = randn(nf1,nf2,nf3).*Csqrt; % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',nk,nf1*nf2);

% Inverse Fourier transform
ktns = reshape(U1*reshape(kh,nf1,[]),nks(1),nf2,nf3); % apply ifft along cols
ktns = permute(ktns,[2 1 3]); % permute so we can deal with rows
ktns = reshape(U2*reshape(ktns,nf2,[]),nks(2),nks(1),nf3); % apply ifft along rows
ktns = permute(ktns,[3 1 2]); % permute so we can deal with slice
ktns = reshape(U3*reshape(ktns,nf3,[]),nks(3),nks(2),nks(1));
ktns = permute(ktns,[3 2 1]);

% view 1st 8 matrices
for j = 1:min(8,nks(3));
    subplot(2,4,j); imagesc(ktns(:,:,j)); title(sprintf('slice %d',j));
end
   
k = ktns(:);  % reshape as a vector


% Make full covariance matrix (for inspection purposes only; will cause
% out-of-memory error if filter dimensions too big!)
C1 = U1*diag(cdiag1)*U1';
C2 = U2*diag(cdiag2)*U2';
C3 = U3*diag(cdiag3)*U3';
Cprior = rho*kron(C3,kron(C2,C1));


%%  Make stimulus and response
nsamps = 1000; % number of stimulus sample
signse = 10;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
subplot(224); % ------
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


%% Compute ASD estimate
fprintf('\n\n...Running ASD_2D...\n');

minlen = 2.5;  % minimum length scale along each dimension

tic; 
[kasd,asdstats] = fastASD(x,y,nks,minlen);
toc;

%%  ---- Make Plots ----

kridge_tns = reshape(kridge,nks);
kasd_tns = reshape(kasd,nks);

for j = 1:min(4,nks(3));
    subplot(3,4,j); imagesc(ktns(:,:,j)); 
    title(sprintf('slice %d',j));
    subplot(3,4,j+4); imagesc(kridge_tns(:,:,j));
    subplot(3,4,j+8); imagesc(kasd_tns(:,:,j));
end
subplot(3,4,1); ylabel('\bf{true k}');
subplot(3,4,5); ylabel('\bf ridge');
subplot(3,4,9); ylabel('\bf ASD');


% Display facts about estimate
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n-----------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f)\n',len(1),asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)\n',rho(1),asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors
err = @(khat)(sum((k-khat(:)).^2)); % Define error function
fprintf('\nErrors:\n------\n  Ridge = %7.2f\n  ASD2D = %7.2f\n\n', ...
     [err(kridge) err(kasd)]);
% 
