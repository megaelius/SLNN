function [al_act, iW] = find_alpha(L,g,g_act,g_ant,w_act,d_act,c1,c2,kmaxBLS,epsal,al_ant,k,wk,d_ant,ialmax,sg_ga2,sg_al0,kmax,isd)
    iW = 4;
    if isd == 7
        ksg = floor(sg_ga2 * kmax);
        al_act = 0.01 * sg_al0;
        if k <= ksg
            al_act = (1 - k/ksg)*sg_al0 + (k/ksg)*al_act;
        end
    else
        if k == 1, alpham = 2;
        elseif ialmax == 1, alpham = al_ant * g_ant' * d_ant / (g_act' * d_act);
        elseif ialmax == 2, alpham = 2 * (L(w_act) - L(wk(:, k - 1))) / (g_act' * d_act);
        end
        [al_act, iW] = uo_BLSNW32(L,g,w_act,d_act,alpham,c1,c2,kmaxBLS,epsal);
    end
end
