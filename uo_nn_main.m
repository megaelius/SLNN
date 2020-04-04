clear;
%
% Parameters for dataset generation
%
num_target = [3];       % Number we want to recognize.
tr_freq    = .5;        % Frequency of the number we want to recongize in the training set.
tr_p       = 250;       % Number of training images.
te_q       = 250;       % Number of test images.
tr_seed    = 123456;    % Seed of the training.
te_seed    = 789101;    % Seed of the test.

%
% Parameters for optimization
%
la = 1.0;                                                     % L2 regularization.
epsG = 10^-6; kmax = 10000;                                   % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 1; icg = 2; irc = 2 ; nu = 1.0;                         % Search direction.
sg_ga1 = 0.05; sg_al0=2; sg_ga2=0.3;                           % stochastic gradient
%
% Optimization
%
t1=clock;                               % Mirar si num_target hace falta -----------------------------------------------------------------------------------
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu);
t2=clock;
fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
uo_nn_Xyplot(Xtr,ytr,wo);
fprintf("niter: %d\n",niter);
%
