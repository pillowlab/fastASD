function [ri,ci] = triuinds(nn,k)
%  [ii] = triuinds(nn,k)
%  [ri,ci] = triuinds(nn,k)
%
%  triuinds(nn,k) - extract row and column indices of upper triangular elements of a matrix
%  of size nn (default k=0 if not provided)
%
%  Inputs:
%   nn - sidelength of square matrix
%    k - which diagonal to start at (0 = main diagonal) (OPTIONAL).
%
%  Outputs:
%   ii - indices of entries of upper triangle (from 1 to nn^2).
%   [ri,ci] - row and column indices of upper triangle

if nargin == 1
    k = 0;
end

if nargout == 1
    ri = find(triu(ones(nn),k));
else
    [ri,ci] = find(triu(ones(nn),k));
end
