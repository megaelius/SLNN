function dk = descent_direction(Xtr, ytr, wk, gL, H, isd, icg, irc, nu, d_ant, k, sg_ga1, tr_p)

    g = @(w) gL(Xtr, ytr, w);
    gw = g(wk(:, k)); betak = 0;

    % Gradient method
    if isd == 1 || (isd == 2 && k == 1), dk = -gw;

    % Conjugate gradient method
    elseif isd == 2
        gprev = g(wk(:, k - 1));

        % If first iteration or must restart betak = 0.
        if ~(irc == 1 && mod(k, length(wk(:, 1))) == 0) && ~(irc == 2 && abs(gw' * gprev / norm(gw)^2) >= nu)
            % Fletcher-Reeves
            if icg == 1, betak = norm(gw)^2 / norm(gprev)^2;
            % Polak-Ribière
        else, betak = max(gw' * (gw - gprev) / norm(gprev)^2, 0);
            end
        end
        dk = -gw + betak * d_ant; % Computation descent direction

    % BFGS
    elseif isd == 3, dk = -H * gw;
    elseif isd == 7
        m = floor(sg_ga1*tr_p);
        aux = randsample(tr_p, m);
        Xtrs = Xtr(:, aux);  ytrs = ytr(aux);
        g = @(w) gL(Xtrs, ytrs, w);
        dk = - 1/m * g(wk(:, k));
    end
end
