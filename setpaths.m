% SETPATHS.m - set paths for fastASD code repository
%
% This script sets the path to include relevant directories for the
% fastASD code package.  You must 'cd' into this directory in order
% to evaluate it.
%
% Github page: https://github.com/pillowlab/fastASD/

basedir = pwd;  % The directory where this script lives
addpath(basedir); % add this directory

% Add a bunch sub-directories (with absoluate path names)
addpath([basedir '/code_fastASD']);
addpath([basedir '/tools_dft']);
addpath([basedir '/tools_kron']);
addpath([basedir '/tools_misc']);
addpath([basedir '/demos']);

