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
    iWk -> Condicions de Wolfe que satisfà:
            0: No satisfà res
            1: Satisfà WC1
            2: Satisfà WC2
            3: Satisfà SWC
    niter -> Numero d'iteracions que fa l'algorisme
%}



function [xk,iWk,niter] = om_uo_solve(x,f,g,epsG,kmax,c1,c2,isd,icg,irc,epsal,kmaxBLS,nu,ialmax)
    n = length(x); k = 1;
    xk = zeros(n, kmax); d_act = zeros(n, 1); H_act = zeros(n, n);
    iWk = zeros(1, kmax); al_act = 0;
    xk(:, 1) = x;  % Initialation vector
    while k < kmax && norm(g(xk(:, k))) > epsG
        d_ant = d_act; H_ant = H_act; al_ant = al_act;
        x_act = xk(:, k);

        % Computation of the descent direction given the method
        d_act = descent_direction(xk, g, H_ant, isd, icg, irc, nu, d_ant, k);

        % Computation of the alfa given the descent direction
        if k == 1, alpham = 2;
        elseif ialmax == 1, alpham = al_ant * g(xk(:, k-1))' * d_ant / (g(x_act)' * d_act);
        elseif ialmax == 2, alpham = 2 * (f(x_act) - f(xk(:, k-1))) / (g(x_act)' * d_act);
        end
        [al_act, iWk(k)] = uo_BLSNW32(f,g,x_act,d_act,alpham,c1,c2,kmaxBLS,epsal);

        % Vector's update
        xk(:, k + 1) = x_act + al_act*d_act;
        if isd == 3
            y = g(xk(:, k + 1)) - g(x_act); s = xk(:, k + 1) - x_act; rhok = 1 / (y' * s);  % Auxiliary variables s, y, rho
            H_act = (eye(n) - rhok*s*y') * H_ant * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
        k = k + 1;
    end
    iWk = iWk(1:k); iWk(k) = NaN; xk = xk(:, 1:k);
    niter = k;
end
