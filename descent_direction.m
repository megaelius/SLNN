function [dk, betak, Hk, tau] = descent_direction(xk, gk, Hk, isd, icg, irc, nu, dk, k, h, delta)

    gx = gk(:, k); betak = 0; hess = h(xk(:, k)); tau = 0;

    % Gradient method
    if isd == 1 || (isd == 2 && k == 1), dk = -gx;

    % Conjugate gradient method
    elseif isd == 2
        gprev = gk(:, k - 1);

        % If first iteration or must restart betak = 0.
        if ~(irc == 1 && mod(k, length(xk(:, 1))) == 0) && ~(irc == 2 && abs(gx' * gprev / norm(gx)^2) >= nu)
            % Fletcher-Reeves
            if icg == 1, betak = norm(gx)^2 / norm(gprev)^2;
            % Polak-Ribière
            else, betak = max(gx' * (gx - gprev) / norm(gprev)^2, 0);
            end
        end
        dk = -gx + betak * dk(:, k - 1); % Computation descent direction

    % BFGS
    elseif isd == 3, dk = -Hk * gx;
end
