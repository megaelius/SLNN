function [dk, g_act] = descent_direction(Xtr, ytr, wk, g, gL, H, isd, icg, irc, nu, d_ant, k, sg_ga1, tr_p)
    g_act = g(wk(:, k)); betak = 0;

    % Gradient method
    if isd == 1 || (isd == 2 && k == 1), dk = -g_act;

    % Conjugate gradient method
    elseif isd == 2
        g_ant = g(wk(:, k - 1));

        % If first iteration or must restart betak = 0.
        if ~(irc == 1 && mod(k, length(wk(:, 1))) == 0) && ~(irc == 2 && abs(g_act' * g_ant / norm(g_act)^2) >= nu)
            % Fletcher-Reeves
            if icg == 1, betak = norm(g_act)^2 / norm(g_ant)^2;
            % Polak-Ribière
        else, betak = max(g_act' * (g_act - g_ant) / norm(g_ant)^2, 0);
            end
        end
        dk = -g_act + betak * d_ant; % Computation descent direction

    % BFGS
    elseif isd == 3, dk = -H * g_act;

    % Stochastic gradient method
    elseif isd == 7
        m = floor(sg_ga1*tr_p);
        aux = randsample(tr_p, m);
        Xtrs = Xtr(:, aux);  ytrs = ytr(aux);
        g = @(w) gL(Xtrs, ytrs, w);
        g_act = g(wk(:, k));
        dk = - 1/m * g_act;
    end
end
