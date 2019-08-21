% varpro_example.m
% Sample problem illustrating the use of varpro.m
%
% Observations y(t) are given at 10 values of t.
%
% The model for this function y(t) is 
%
%   eta(t) = c1 exp(-alpha2 t)*cos(alpha3 t) 
%          + c2 exp(-alpha1 t)*cos(alpha2 t)
%
% The linear parameters are c1 and c2, and the 
%  nonlinear parameters are alpha1, alpha2, and alpha3.
%
% The two nonlinear functions in the model are
%
%   Phi1(alpha,t) = exp(-alpha2 t)*cos(alpha3 t),
%   Phi2(alpha,t) = exp(-alpha1 t)*cos(alpha2 t).
%
% Dianne P. O'Leary and Bert W. Rust, September 2010

%disp('****************************************')
%disp('Sample problem illustrating the use of varpro.m')

% Data observations y(t) were taken at these times:

%t = [0;.1;.22;.31;.46;.50;.63;.78;.85;.97];

%y = [ 6.9842;  5.1851;  2.8907;  1.4199; -0.2473; 
%     -0.5243; -1.0156; -1.0260; -0.9165; -0.6805];
load INPUT
%y = ft';
%t = t';
rt = [t linspace(min(t),max(t),100)];
y = interp1(t,ft,rt,'spline')';
w = [ones(size(t)) .01*ones(1,100)]';
rt = rt';

% The weights for the least squares fit are stored in w.

% Set varpro options to display the progress of the iteration
% and to check the correctness of the derivative formulas.

options = optimset('Display','iter','DerivativeCheck','on');

% Set the initial guess for alpha and call varpro to estimate
% alpha and c.

alphainit = exp(-3:-1)';

tic
[alpha,c,wresid,resid_norm,y_est,Regression] = ...
     varpro(y,w,alphainit,3,@(alpha)adaex(alpha,rt),[],[],options);
toc

save( 'OUTPUT', 'alpha','c','wresid','resid_norm','y_est' );
