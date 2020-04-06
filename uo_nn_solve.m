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
    L = @(X, Y, w) norm(y(X, w) - Y)^2 + (la * norm(w)^2)/2;
    gL = @(X, Y, w) 2 * sig(X) * ((y(X, w) - Y) .* y(X, w) .* (1 - y(X, w)))' + la * w;
    g = @(w) gL(Xtr, ytr, w);
    acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/size(Xds,2);

    n = length(w0); k = 1;
    wk = zeros(n, kmax); d_act = zeros(n, 1); H_act = eye(n);
    iWk = zeros(1, kmax); al_act = 0;
    wk(:, 1) = w0;  % Initialation vector
    
    t1 = clock;
    while k < kmax && norm(gL(Xtr,ytr,wk(:, k))) > epsG
        d_ant = d_act; H_ant = H_act; al_ant = al_act;
        w_act = wk(:, k);

        % Computation of the descent direction given the method
        d_act = descent_direction(Xtr, ytr, wk, gL, H_ant, isd, icg, irc, nu, d_ant, k, sg_ga1, tr_p);

        % Computation of the alfa given the descent direction
        if k == 1, alpham = 2;
        elseif ialmax == 1, alpham = al_ant * g(wk(:, k-1))' * d_ant / (g(w_act)' * d_act);
        elseif ialmax == 2, alpham = 2 * (L(Xtr, ytr, w_act) - L(Xtr, ytr, wk(:, k-1))) / (g(w_act)' * d_act);
        end

        if isd == 7
            ksg = floor(sg_ga2*kmax);
            al_act = 0.01*sg_al0;
            if k <= ksg
                al_act = (1 - k/ksg)*sg_al0 + (k/ksg)*al_act;
            end
        else, [al_act, iWk(k)] = uo_BLSNW32(@(w) L(Xtr, ytr, w),g,w_act,d_act,alpham,c1,c2,kmaxBLS,epsal);
        end

        % Vector's update
        wk(:, k + 1) = w_act + al_act*d_act;
        if isd == 3
            y = g(wk(:, k + 1)) - g(w_act); s = wk(:, k + 1) - w_act; rhok = 1 / (y' * s);  % Auxiliary variables s, y, rho
            H_act = (eye(n) - rhok*s*y') * H_ant * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
        fprintf("k = %d, acc = %f , norm = %f, al = %f\n",k,acc(Xtr,ytr,wk(:, k + 1)),norm(gL(Xtr,ytr,wk(:, k+1))),al_act); 
        k = k + 1;
    end
    t2 = clock;
    
    iWk = iWk(1:k); iWk(k) = NaN; wk = wk(:, 1:k);
    niter = k;
    wo = wk(:, length(wk));
    tex = etime(t2,t1);
    tr_acc = acc(Xtr,ytr,wo); te_acc = acc(Xte,yte,wo);
    fo = L(Xtr, ytr, wo);
end
