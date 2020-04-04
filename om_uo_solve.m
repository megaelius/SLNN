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



function [xk,iWk,niter] = om_uo_solve(x,f,g,epsG,kmax,c1,c2,isd,icg,irc,epsal,kmaxBLS,nu)
    n = length(x); k = 1;
    xk = zeros(n, kmax); dk = zeros(n, kmax); Hk = zeros(n, n, kmax);
    iWk = zeros(1, kmax);
    xk(:, 1) = x; Hk(:, :, 1) = eye(n);  % Initialation vector
    while k < kmax && norm(g(xk(:, k))) > epsG
        xact = xk(:, k);
        % Computation of the descent direction given the method
        dk(:, k) = descent_direction(xk, g, Hk(:, :, k), isd, icg, irc, nu, dk, k);
        % Computation of the alfa given the descent direction
        alpham = 2;%2 * (f(xact) - f(xk(:, k - 1))) / (g(xact)' * dk(:, k))
        [al, iWk(k)] = uo_BLSNW32(f,g,xact,dk(:, k),alpham,c1,c2,kmaxBLS,epsal);
        % Vector's update
        xk(:, k + 1) = xact + al*dk(:, k);
        if isd == 3
            y = g(xk(:, k + 1)) - g(xact); s = xk(:, k + 1) - xact; rhok = 1 / (y' * s);  % Auxiliary variables s, y, rho
            Hk(:, :, k + 1) = (eye(n) - rhok*s*y') * Hk(:, :, k) * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
        k = k + 1;
    end
    iWk = iWk(1:k); iWk(k) = NaN; xk = xk(:, 1:k);
    niter = k;
end
