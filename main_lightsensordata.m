% Script for light sensor data experiments 
% produces cumulative regret after T rounds

T = 10^3;

load('iw_data_all.mat');
rawtraindata = data.train.sensordata;
rawtestdata = data.test.sensordata;
trainsize = size(rawtraindata,1);
mean_data = mean(rawtraindata);
sigma_data = std(rawtraindata);
noise = (rawtraindata - repmat(mean_data,trainsize,1))./repmat(sigma_data,trainsize,1);
K = cov(noise);
f_test = bsxfun(@rdivide,mean(rawtestdata),max(mean(rawtestdata)));
B = max(mean(rawtestdata));

noise_index = 3; % denotes real life heavy-tailed data
noise_param = rawtestdata; % heavy-tailed payoffs
alpha = 1; 
v = mean(rawtestdata(:).^2);

lambda = 1;
tgp_ucb_regret = tgp_ucb(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v);

epsilon = 0.1;
ata_gp_ucb_nystrom_regret = ata_gp_ucb_nystrom(B,f_test,K,T,lambda,noise_index,noise_param,alpha,v,epsilon);

points = data.coords;
n = size(points,1);
r = range(points);
m = min(points);
sensor_location = (points - repmat(m,n,1))./repmat(r,n,1);
l = sqrt(0.1);
m_bar = 16;
Phi_qff = qff_approx(sensor_location',l,m_bar);
ata_gp_ucb_qff_regret = ata_gp_ucb_qff(B,f_test,Phi_qff,T,lambda,noise_index,noise_param,alpha,v);





