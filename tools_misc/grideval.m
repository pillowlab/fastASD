function [fvals,gridvecs] = grideval(ngrid,gridranges,fptr)
% GRIDEVAL - evaluates a function on a grid of points
%
% [fvals,gridvecs] = grideval(ngrid,gridranges,fptr)
%
% INPUTS
% ------
% ngrid [1 x 1] - number of grid points
% gridgranges [n x 2] specify range for each dimension of grid
% fptr - handle for function to evaluate
%
% OUTPUTS
% -------
% fvals - full grid of function values
% gridvecs - array whose columns are grid coordinate vectors along each dimension
%
% Updated 2015.01.29 (jwp)

% Set grid 
ndim = size(gridranges,1); % number of dimensions in grid
gridvecs = zeros(ngrid,ndim);
for jdim = 1:ndim
    len = diff(gridranges(jdim,:)); % length of interval
    endpts = gridranges(jdim,:)+len/(2*ngrid)*[1 -1]; % endpoints for grid
    gridvecs(:,jdim) = linspace(endpts(1),endpts(2),ngrid);
end

% Evaluate function on grid
siz = ngrid*ones(1,ndim); % size of grid
npts = prod(siz); % total number of points in grid
fvals = zeros(siz); % initialize space for grid values
for idx = 1:npts
    vinds = ind2subv(siz,idx); % indices for grid
    gridpt = gridvecs(sparse(vinds,1:ndim,true,ngrid,ndim)); % make input vector
    fvals(idx) = fptr(gridpt); % evaluate
end

    
    
