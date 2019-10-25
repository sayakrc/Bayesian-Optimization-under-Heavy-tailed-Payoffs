% Script for synthetic data experiments 
% produces cumulative regret after T rounds

T = 10^3;

data = 0.01:0.01:1;
[d,arms] = size(data);
kernel_index = 1;   % use 1 for SE kernel and 2 for Matern kernel
l = 0.2;
K = zeros(arms);
for i = 1 : arms
    for j = 1 : arms
        x = data(:,i);
        y = data(:,j);
        K(i,j) = kernel_function(x,y,l,kernel_index);
    end
end

rkhs_index = 1;  % use 1 for RKHS functions with coefficients in [-1,1] and 2 for coefficients in [0,1]
p = 100;
f = gen_rkhs_function(data,l,p,rkhs_index,kernel_index);
f_test = bsxfun(@rdivide,f,max(abs(f)));
B = max(abs(f_test));

noise_index = 1; % use 1 for Students-t distribution and 2 for Pareto distribution
if noise_index == 1
    noise_param = 3; % degrees of freedom of the Students-t distribution
    alpha = 1;
    v = noise_param/(noise_param-2)+B^2;
else
    noise_param = 2; % shape paremeter of Pareto distribution
    alpha = 0.9;
    v = (B^(1+alpha))/((2^alpha)*(1-alpha));
end

lambda = 1;
tgp_ucb_regret = tgp_ucb(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v);

epsilon = 0.1;
ata_gp_ucb_nystrom_regret = ata_gp_ucb_nystrom(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v,epsilon);

m_bar = 32;
Phi_qff = qff_approx(data,l,m_bar);
ata_gp_ucb_qff_regret = ata_gp_ucb_qff(B,f_test,Phi_qff,T,lambda,noise_index,noise_param,alpha,v);




