T = 10^3;

data = 0.01:0.01:1;
[d,arms] = size(data);
kernel_index = 1;
l = 0.2;
K = zeros(arms);
for i = 1 : arms
    for j = 1 : arms
        x = data(:,i);
        y = data(:,j);
        K(i,j) = kernel_function(x,y,l,kernel_index);
    end
end

rkhs_index = 1;
p = 100;
f = gen_rkhs_function(data,l,p,rkhs_index,kernel_index);
f_test = bsxfun(@rdivide,f,max(abs(f)));
B = max(abs(f_test));

noise_index = 1;
if noise_index == 1
    noise_param = 3;
    alpha = 1;
    v = noise_param/(noise_param-2)+B^2;
else
    noise_param = 2;
    alpha = 0.9;
    v = (B^(1+alpha))/((2^alpha)*(1-alpha));
end

lambda = 1;
tgp_ucb_regret = tgp_ucb(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v);

eps = 0.1;
ata_gp_ucb_nystrom_regret = ata_gp_ucb_nystrom(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v,eps);

qff_features = 32;
Phi_qff = qff_approx(data,l,qff_features);
ata_gp_ucb_qff_regret = ata_gp_ucb_qff(B,f_test,Phi_qff,T,lambda,noise_index,noise_param,alpha,v);




