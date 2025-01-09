% Input data
data = csvread('Datasets/Control.csv');

% Obtain data size and true annotations
m = size(data, 2);
X = data(:,1:m-1);
X = mapminmax(X',0,1)';
ref = data(:,m);
clus_num = length(unique(ref));

% Perform the KM+LoDD algorithm
clus = ModKmeans(X,'k_num',20,'ratio',0.2,'NumClus',clus_num,'Method','lodd');
ACC = getACC(ref,clus);
NMI = getNMI(ref,clus);
disp(['ACC:', num2str(ACC), ' NMI:', num2str(NMI)]);