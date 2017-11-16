function [B,wvec,wcos,wsin] = realnufftbasis(tvec,T,N)
% Real basis for non-uniform discrete fourier transform 
%
% [B,wvec] = realnufftbasis(tvec,T,N)
%
% Makes basis of sines and cosines B for a function sampled at nonuniform
% time points in tvec, with a circular boundary on [0, T], and spacing of
% frequencies given by integers m*(2pi/T) for m \in [-T/2:T/2].
%
% If maxw input given, only returns frequencies with abs value < maxw
%
% INPUTS:
%  tvec [nt x 1] - column vector of non-uniform time points for signal 
%     T  [1 x 1] - circular boundary for function in time 
%     N  [1 x 1] - number of Fourier frequencies to use
%
% OUTPUTS:
%      B [ni x N] - basis matrix for mapping Fourier representation to real
%                   points on lattice of points in tvec.
%    wvec [N x 1] - DFT frequencies associated with columns of B
%   
% See also: realfft, realifft

% make column vec if necessary
if size(tvec,1) == 1
    tvec = tvec';
end

if max(tvec+1e-6)>T
    error('max(tvec) greater than circular boundary T!');
end

% Make frequency vector
ncos =  ceil((N+1)/2); % number of cosine terms (positive freqs)
nsin = floor((N-1)/2); % number of sine terms (negative freqs)
wcos = (0:ncos-1)'; % cosine freqs
wsin = (-nsin:-1)'; % sine freqs
wvec = [wcos;wsin]; % complete frequency vector for realfft representation

if nsin > 0
    B = [cos((2*pi/T)*wcos*tvec'); sin((2*pi/T)*wsin*tvec')]/sqrt(T/2);
else
    B = cos((2*pi/T)*wcos*tvec')/sqrt(T/2);
end

% make DC term into a unit vector
B(1,:) = B(1,:)/sqrt(2);  

% if N is even, make Nyquist (highest-freq cosine) term into unit vector
if (mod(N,2)==0) && (N==T) 
    B(ncos,:) = B(ncos,:)/sqrt(2);
end
