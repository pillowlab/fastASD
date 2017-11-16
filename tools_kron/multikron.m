function B = multikron(A)
% Form matrix (A{n} kron ... A{2} kron A{1})
%
% B = multikron(A)
%
% Input: cell array of matrices, ordered in reverse order of kron operation
%
% Output: giant kronecker matrix

if length(A) == 1
    B = A{1};
elseif length(A) == 2
    B = kron(A{2},A{1});
else
    B = kron(multikron(A(2:end)),A{1});
end
