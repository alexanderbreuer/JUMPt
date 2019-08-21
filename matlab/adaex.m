
function [Phi,dPhi,Ind] = adaex(alpha,t)

% function [Phi,dPhi,Ind] = adaex(alpha,t)
% This is a sample user-defined function to be used by varpro.m.
%
% The model for this sample problem is
%
%   eta(t) = c1 exp(-alpha1 t)
%          + c2 exp(-alpha2 t)
%          + c2 exp(-alpha3 t)
%          + c2 exp(-alpha4 t)
%          = c1 Phi1 + c2 Phi2 + c3 Phi3 + c4 Phi4.
%
% Given t and alpha, we evaluate Phi, dPhi, and Ind.
%
% Dianne P. O'Leary and Bert W. Rust, September 2010.


% Evaluate Phi1 = exp(-alpha2 t)*cos(alpha3 t),
%          Phi2 = exp(-alpha1 t)*cos(alpha2 t),
% at each of the data points in t.

    Phi = exp(abs(alpha)*-t')';


% The nonzero partial derivatives of Phi with respect to alpha are
%              d Phi_1 / d alpha_2 ,
%              d Phi_1 / d alpha_3 ,
%              d Phi_2 / d alpha_1 ,
%              d Phi_2 / d alpha_2 ,
% and this determines Ind.
% The ordering of the columns of Ind is arbitrary but must match dPhi.

    Ind = [1 2 3  
           1 2 3 ];

% Evaluate the four nonzero partial derivatives of Phi at each of 
% the data points and store them in dPhi.
    dPhi = (((alpha./abs(alpha))*-t') .* Phi')';

