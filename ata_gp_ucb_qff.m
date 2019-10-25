function cumulative_regret = ata_gp_ucb_qff(B,f_star,Phi,T,lambda,n_ind,n_param,alpha,v)

        m = size(Phi,1);
        [~, arms] = size(f_star);
        [opt_value,~] = max(f_star);
        reward_so_far = zeros(1,T);
        Phi_so_far = zeros(m,T);
        V = lambda*eye(m);  
        cumulative_regret = 0;
            
        for round = 1 : T
            
            b_t = round^((1-alpha)/(2*(1+alpha)))*(v/log(m*T))^(1/(1+alpha));
            beta_t = B*sqrt(lambda) + sqrt(m)*v^(1/(1+alpha))*(log(m*T))^(alpha/(1+alpha))*round^((1-alpha)/(2*(1+alpha)));
           
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
                
            reward_so_far(round) = reward_obt;
            phi = Phi(:,arm_pld);
            Phi_so_far(:,round) = phi;
            V = V + phi*phi';
                
            M = V^(-0.5);
            C1 = Phi_so_far(:,1:round);
            C = M*C1;
            y = reward_so_far(1:round);
            r = zeros(m,1);
            for i = 1 : m
                u = C(i,:);
                w = u.*y;
                r(i) = sum(w(abs(w) <= b_t));
            end
            theta = M*r;
            mean_vec = Phi'*theta;
            a = M*Phi;
            var_vec = diag(a'*a);
        end
end