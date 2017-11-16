function y = kronmult(Amats,x,ii)
% Multiply matrix (.... A{3} kron A{2} kron A{1})(:,ii) by x
%
% y = kronmult(Amats,x,ii);
% 
% INPUT
%   Amats  - cell array with matrices {A1, ..., An}
%        x - matrix to multiply with kronecker matrix formed from Amats
%      ii  - binary vector indicating sparse locations of x rows (OPTIONAL)
%
% OUTPUT
%    y - matrix (with same number of cols as x)
%
% Equivalent to (for 3rd-order example)
%    y = (A3 \kron A2 \kron A1) * x
% or in matlab:
%    y = kron(A3,kron(A2,A1)))*x
%
% Exploits the identity that 
%    y = (A2 kron A1) * x 
% is the same as
%    y = vec( A1 * reshape(x,m,n) * A2' )
% but the latter is far more efficient.
%
% Computational cost: 
%    Given A1 [p x n] and A2 [q x m], and x a vector of length nm, 
%    standard implementation y = kron(A2,A1)*x costs O(nmpq)
%    whereas this algorithm costs O(nm(p+q))

ncols = size(x,2);

% Check if 'ii' indices passed in for inserting x into larger vector
if nargin > 2
    x0 = zeros(length(ii),ncols);
    x0(ii,:) = x;
    x = x0;
end
nrows = size(x,1);

% Number of matrices
nA = length(Amats);

if nA == 1
    % If only 1 matrix, standard matrix multiply
    y = Amats{1}*x;
else
    % Perform remaining matrix multiplies
    y = x; % initialize y with x
    for jj = 1:nA
        [ni,nj] = size(Amats{jj}); %
        y = Amats{jj}*reshape(y,nj,[]); % reshape & multiply
        y =  permute(reshape(y,ni,nrows/nj,[]),[2 1 3]); % send cols to 3rd dim & permute
        nrows = ni*nrows/nj; % update number of rows after matrix multiply
    end
    
    % reshape to column vector
    y = reshape(y,nrows,ncols);

end

