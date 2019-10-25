function cumulative_regret = tgp_ucb(B,f_star,K,T,lambda,n_ind,n_param,alpha,v)
 
        [~, arms] = size(f_star);
        [opt_value,~] = max(f_star);
        cov_mat = K;
        mean_vec = zeros(arms,1);
        cumulative_regret = 0;
            
            
        for round = 1 : T
            
            b_t = round^(1/(2*(1+alpha)))*v^(1/(1+alpha));
            beta_t = B*sqrt(lambda) + round^(1/(2*(1+alpha)))*v^(1/(1+alpha))*sqrt(log(round)); 
            if round == 1
                arm_pld = randi(arms);
            else
                [~,arm_pld] = max(mean_vec + beta_t*sqrt(var_vec));
            end
            cumulative_regret = cumulative_regret + opt_value - f_star(arm_pld);
            
            if n_ind == 1
                nu = n_param;
                reward_obt =  f_star(arm_pld) + trnd(nu);
            elseif n_ind == 2
                kval = 1/n_param;
                thta = (n_param-1)*f_star(arm_pld)/n_param;
                sgma = thta/n_param;
                reward_obt = gprnd(kval,sgma,thta);
            else
                reward_obt = n_param(randi(size(n_param,1)),arm_pld);
            end
            
            if abs(reward_obt) > b_t
                reward_obt = 0;
            end
            [mean_vec,cov_mat] = posterior_update(mean_vec,cov_mat,reward_obt,arm_pld,lambda);
            var_vec = (1/lambda)*diag(cov_mat);
            
        end
    
end