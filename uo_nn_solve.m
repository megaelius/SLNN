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
    k = 0;
    while k < kmax && norm(g(w)) > epsG
        % Calculo de la dirección de descenso
    end


    while k < kmax && norm(gk(:, k)) > epsG
        % Computation of the descent direction given the method
        [dk(:, k), Hk(:, :, k)] = descent_direction(xk, gk, Hk(:, :, k), isd, icg, irc, nu, dk, k, h, delta);
        % Computation of the alfa given the descent direction
        if isd == 4, alk(k) = 1; iWk(k) = 4;
        else, [alk(k), iWk(k)] = find_alfa(xk(:, k), dk(:, k), f, g, almin, almax, rho, c1, c2, iW, Hk(:, :, k));
        end

        % Vector's update
        xk(:, k + 1) = xk(:, k) + alk(k)*dk(:, k);
        gk(:, k + 1) = g(xk(:, k + 1));
        if isd == 3
            y = gk(:, k + 1) - gk(:, k); s = xk(:, k + 1) - xk(:, k); rhok = 1 / (y' * s);  % Auxiliary variables s, y, rho
            Hk(:, :, k + 1) = (eye(n) - rhok*s*y') * Hk(:, :, k) * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
        k = k + 1;
    end
    alk = alk(1:k); iWk = iWk(1:k); iWk(k) = NaN; xk = xk(:, 1:k); dk = dk(:, 1:k); Hk = Hk(:, :, 1:k); tauk = tauk(1:k);

end
