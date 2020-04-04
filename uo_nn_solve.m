% wo = w óptima, fo = valor de la función en el mínimo, tr_acc = precisión de training
% te_acc = precisión de test, niter = número de iteraciones, tex = tiempo de ejecución después de generar el dataset.
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu)
    %
    % Generación de datasets
    %
    [Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, tr_freq);  % La frecuencia en el test es la misma que en el training -> Tiene que ser asi?? ---------


    w = zeros(35, 1); %punto inicial
    L = @(w) norm(y(Xtr,w)-ytr)^2 + (la*norm(w)^2)/2;
    gL = @(w) 2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))'+la*w;
    [wk,iWk,niter] = om_uo_solve(w,L,gL,epsG,kmax,c1,c2,isd,icg,irc,epsal,kmaxBLS);

end
