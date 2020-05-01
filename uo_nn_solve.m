% wo = w óptima, fo = valor de la función en el mínimo, tr_acc = precisión de training
% te_acc = precisión de test, niter = número de iteraciones, tex = tiempo de ejecución después de generar el dataset.
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu)
    %
    % Generación de datasets
    %
    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, tr_freq);  % La frecuencia en el test es la misma que en el training -> Tiene que ser asi?? ---------

    w0 = zeros(35, 1); %punto inicial
    sig = @(X) 1 ./ (1 + exp(-X));
    y = @(X, w) sig(w' * sig(X));
    L = @(w) norm(y(Xtr, w) - ytr)^2 + (la * norm(w)^2)/2;
    gL = @(X, Y, w) 2 * sig(X) * ((y(X, w) - Y) .* y(X, w) .* (1 - y(X, w)))' + la * w;
    g = @(w) gL(Xtr, ytr, w);
    acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/size(Xds,2); %accuracy
    
    %g_ant is initialize to 1 only for entering the while, this value is
    %never used
    n = length(w0); k = 1; g_act = 1; 
    wk = zeros(n, kmax); d_act = zeros(n, 1); H_act = eye(n);
    iWk = zeros(1, kmax); al_act = 0;
    wk(:, 1) = w0;
    t1 = clock;
    while k < kmax && norm(g_act) > epsG
        d_ant = d_act; H_ant = H_act; al_ant = al_act;
        w_act = wk(:, k);g_ant = g_act;

        % Computation of the descent direction given the method
        [d_act, g_act] = descent_direction(Xtr, ytr, wk, g, gL, H_ant, isd, icg, irc, nu, d_ant, k, sg_ga1, tr_p);

        % Computation of the alfa given the descent direction
        [al_act, iWk(k)] = find_alpha(L,g,g_act,g_ant,w_act,d_act,c1,c2,kmaxBLS,epsal,al_ant,k,wk,d_ant,ialmax,sg_ga2,sg_al0,kmax,isd);

        % Vector's update
        wk(:, k + 1) = w_act + al_act*d_act;
        if isd == 3
            y = g(wk(:, k + 1)) - g_act; s = wk(:, k + 1) - w_act; rhok = 1 / (y' * s);  % Auxiliary variables s, y, rho
            H_act = (eye(n) - rhok*s*y') * H_ant * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
        k = k + 1;
    end
    t2 = clock;
    iWk = iWk(1:k); iWk(k) = NaN; wk = wk(:, 1:k);
    niter = k;
    wo = wk(:, k);
    tex = etime(t2,t1);
    tr_acc = acc(Xtr,ytr,wo); te_acc = acc(Xte,yte,wo);
    fo = L(wo);
end
