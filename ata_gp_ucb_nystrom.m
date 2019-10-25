function cumulative_regret = ata_gp_ucb_nystrom(B,f_star,K,T,lambda,n_ind,n_param,alpha,v,eps)

        q = (1/eps^2)*log(T);
        [~, arms] = size(f_star);
        [opt_value, ~] = max(f_star);
        reward_so_far = zeros(1,T);
        arm_plays = false(1,arms);
        play_count = zeros(1,arms);
        Q = zeros(arms,T);
        var_vec = diag(K);
        cumulative_regret = 0;
                
        for round = 1 : T
           
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
            arm_plays(arm_pld) = true;
            play_count(arm_pld) = play_count(arm_pld) + 1;
            Q(arm_pld,play_count(arm_pld)) = round;
            arm_so_far = find(arm_plays);
            len = length(arm_so_far);
            
            D = false(1,arms);
            m = 0;
            for i = 1 : len
                temp = arm_so_far(i);
                p = min(q*var_vec(temp),1);
                z = binornd(1,p);
                if z == 1
                    D(temp) = true;
                    m = m + 1;
                end
            end
            b_t = round^((1-alpha)/(2*(1+alpha)))*(v/log((m+1)*T))^(1/(1+alpha));
            beta_t = B*sqrt(lambda) + sqrt(m)*v^(1/(1+alpha))*(log((m+1)*T))^(alpha/(1+alpha))*round^((1-alpha)/(2*(1+alpha)));
           
            k_D = K(D,1:arms,1);
            K_D = K(D,D,1);
            A = pinv(K_D^0.5);
            Phi = A*k_D;
    
            V = lambda*eye(m);
            Phi_so_far = zeros(m,round);
            for i = 1 : len
                temp1 = arm_so_far(i);
                temp2 = Q(temp1,1:play_count(temp1));
                phi = Phi(:,temp1);
                Phi_so_far(:,temp2) = repmat(phi,[1,play_count(temp1)]);
                V = V + play_count(temp1)*(phi*phi');
            end
            
            M = V^(-0.5);
            C = M*Phi_so_far;
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
            var_vec = (1/lambda)*(diag(K) - diag(Phi'*Phi)) + diag(a'*a);
        end
end