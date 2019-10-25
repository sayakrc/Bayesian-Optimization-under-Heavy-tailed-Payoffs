% Script for stock market data experiments 
% produces cumulative regret after T rounds

T = 10^3;

fin = importfile('fin.csv', 2, 824);
data = fin(:,all(~isnan(fin)));
[days, stocks] = size(data);
mean_stocks = mean(data);
sigma_stocks = std(data);
noise = (data - repmat(mean_stocks,days,1))./repmat(sigma_stocks,days,1);

f_test = bsxfun(@rdivide,mean_stocks,max(mean_stocks));
K = cov(noise);
B = max(mean_stocks);

noise_index = 3; % denotes real life heavy-tailed data
noise_param = data; % heavy-tailed payoffs
alpha = 1; 
v = mean(data(:).^2);

lambda = 1;
tgp_ucb_regret = tgp_ucb(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v);

epsilon = 0.1;
ata_gp_ucb_nystrom_regret = ata_gp_ucb_nystrom(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v,epsilon);









 
   