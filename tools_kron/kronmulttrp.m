function y = kronmulttrp(Amats,varargin)
% Multiply matrix (A{2} kron A{1})^T times vector x 
%
% y = kronmulttrp(Amats,x,ii);
% 
% Computes:
% y = (A_n  kron A_{n-1}  ... A_2 kron  A_1)^T x, 
%   = (A_n' kron A_{n-1}' ... A_2' kron A_1') x
%
% See 'kronmult' for details

Amats = cellfun(@transpose,Amats,'UniformOutput',false);
y = kronmult(Amats,varargin{:});
