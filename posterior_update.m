function [mu_new,cov_new] = posterior_update(mu_old,cov_old,y_t,x_t,lambda)

    
    arms = length(mu_old);
    a = cov_old(:,x_t);
    b = cov_old(x_t,:);
    c = lambda + cov_old(x_t,x_t);
    d = y_t-mu_old(x_t);

    A = repmat(a,1,arms);
    B = repmat(b,arms,1);

    mu_new = mu_old + (d/c)*a;
    cov_new = cov_old - (1/c)*(A.*B);
end


