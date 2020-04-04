%{
 ----------------------- Variables de entrada ---------------------------
Generals:
    x -> punt actual
    f, g, i h -> funció, primera i segona derivada respect.
    epsG -> Tolerància / Epsilon
    kmax -> Nombre màxim d'iteracions
    almax, almin -> alfa maxima i mínima
    rho -> Disminució de l'alfa per a l'algorisme uo_BLS
    c1, c2 -> Wolfe Conditions
    iW -> Mètode alfa
        0: Exact line search
        1: WC
        2: SWC
    isd -> Mètode per trobar la search direction
        1: GM
        2: CGM
        3: BFGS
        4: NM
        5: MNM-SD
        6: MNM-CMI

CGM:
    icg -> variant
        1: Fletcher-Reeves
        2: Polak-Ribière +
    irc -> restart condition
        0: No restart
        1: RC1
        2: RC2
    v -> nu per a la RC2




----------------------- Variables de sortida ----------------------------
Generals:
    xk -> Vector de posicions
    dk -> Vector de direccions de descens
    alk -> Vector de alfes
    iWk -> Condicions de Wolfe que satisfà:
            0: No satisfà res
            1: Satisfà WC1
            2: Satisfà WC2
            3: Satisfà SWC
%}



function [xk,dk,gk,alk,iWk,betak,Hk,tauk] = om_uo_solve(x,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta)
    n = length(x); k = 1;
    xk = zeros(n, kmax); dk = zeros(n, kmax); Hk = zeros(n, n, kmax);
    iWk = zeros(1, kmax);
    xk(:, 1) = x; Hk(:, :, 1) = eye(n);  % Initialation vector
    while k < kmax && norm(gk(:, k)) > epsG
        % Computation of the descent direction given the method
        [dk(:, k), Hk(:, :, k)] = descent_direction(xk, gk, Hk(:, :, k), isd, icg, irc, nu, dk, k, h, delta);
        % Computation of the alfa given the descent direction
        if isd == 4, alk(k) = 1; iWk(k) = 4;
        else, [al, iWk(k)] = find_alfa(xk(:, k), dk(:, k), f, g, almin, almax, rho, c1, c2, iW, Hk(:, :, k));
        end

        % Vector's update
        xk(:, k + 1) = xk(:, k) + al*dk(:, k);
        if isd == 3
            y = g(xk(:, k + 1)) - g(xk(:, k)); s = xk(:, k + 1) - xk(:, k); rhok = 1 / (y' * s);  % Auxiliary variables s, y, rho
            Hk(:, :, k + 1) = (eye(n) - rhok*s*y') * Hk(:, :, k) * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
        k = k + 1;
    end
    iWk = iWk(1:k); iWk(k) = NaN; xk = xk(:, 1:k); dk = dk(:, 1:k); Hk = Hk(:, :, 1:k);
end















%%%%%%%%%%%%%%



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
    elseif isd == 4, dk = - inv(hess) * gx;
    elseif isd == 5
        [V, D] = eig(hess);

        for i = 1:length(hess)
            D(i, i) = max(D(i, i), delta);
        end

        Hk = V * D * V';
        dk = -inv(Hk) * gx;
    elseif isd == 6
        laUB = norm(hess, 'fro');
        i = 0;
        [~, p] = chol(hess + tau * eye(length(hess)));

        while p > 0
            tau = (1.01 - 1/2^i)*laUB;
            i = i + 1;
            [~, p] = chol(hess + tau * eye(length(hess)));
        end

        Hk = hess + tau * eye(length(hess));
        dk = -inv(Hk) * gx;
    end
end
