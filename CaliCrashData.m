clear;
% CaliCrashDataSubset1 has already been sorted by Date first, then Time.
tCali = readtable("CaliCrashDataSubset1.csv");

% For the purposes of this dataset, March 19th, 2020 is the cutoff date
% for Pre Covid, given that it is the day a statewide shelter-in-place order was issued.
% https://calmatters.org/health/coronavirus/2021/03/timeline-california-pandemic-year-key-points/

PreCovid = tCali(1:1045364, :);
PostCovid = tCali(1045365:end, :);

% Population estimates from https://www.census.gov/quickfacts/CA
% Population of California as of July 1, 2019 is: 39,512,223

PreCovid.killed_victims = PreCovid.killed_victims / 39512223;
PreCovid.injured_victims = PreCovid.injured_victims / 39512223;
PreCovid = convertvars(PreCovid, {'county_location'}, 'categorical');

PreCovid.county_location = grp2idx(PreCovid.county_location);

PostCovid.killed_victims = PostCovid.killed_victims / 39512223;
PostCovid.injured_victims = PostCovid.injured_victims / 39512223;
PostCovid = convertvars(PostCovid, {'county_location'}, 'categorical');

Loc = unique(PostCovid.county_location);
LocID = grp2idx(Loc);
Loc = array2table(Loc);
LocID = array2table(LocID);
Loc(:,2) = LocID(:, 1);

PostCovid.county_location = grp2idx(PostCovid.county_location);


[s1, f1] = size(PreCovid);


% Scatter Plots for Multivariate stuff, excluding date and time.
labels = {'date', 'time', 'county', 'alcohol involved', 'killed victims', 'injured victims', 'party count', 'highway status'};
for i = 3:f1
    for j = 3:f1
        if i < j && j ~= 1 && j ~= 2
            x = table2array(PreCovid(:, i));
            y = table2array(PreCovid(:, j));
            x2 = table2array(PostCovid(:, i));
            y2 = table2array(PostCovid(:, j));
            figure;
            scatter(x, y);
            hold on;
            scatter(x2, y2);
            xlabel(labels(:, i));
            ylabel(labels(:, j));
            legend({'Pre-Covid', 'Post-Covid'});
        end
    end
end

% AS ALCOHOL INVOLVED AND HIGHWAY INDICATORS ARE BINARY VALUES, 
% IT IS IMPOSSIBLE TO FIND A NORMAL DISTRIBUTION FOR THOSE FEATURES

figure
[mu, std] = normfit(PreCovid.county_location);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'b');
hold on
[mu, std] = normfit(PostCovid.county_location);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'r');
title('Normal Distribution of County')
legend({'Pre-Covid', 'Post-Covid'});
hold off

figure
[mu, std] = normfit(PreCovid.killed_victims);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'b');
hold on
[mu, std] = normfit(PostCovid.killed_victims);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'r');
title('Normal Distribution of Deaths')
legend({'Pre-Covid', 'Post-Covid'});
hold off

figure
[mu, std] = normfit(PreCovid.injured_victims);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'b');
hold on
[mu, std] = normfit(PostCovid.injured_victims);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'r');
title('Normal Distribution of Injuries')
legend({'Pre-Covid', 'Post-Covid'});
hold off

figure
[mu, std] = normfit(PreCovid.party_count);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'b');
hold on
[mu, std] = normfit(PostCovid.party_count);
xKilled = linspace(mu - 3 * std, mu + 3 * std, 100);
plot(xKilled, normpdf(xKilled, mu, std),'r');
title('Normal Distribution of Involved Parties')
legend({'Pre-Covid', 'Post-Covid'});
hold off


% Add Features that simplify the date to pre or post COVID
tCali(1:1045364, 9) = {0};
tCali(1045365:end, 9) = {1};

% Remove observations with missing elements
tCali = rmmissing(tCali);
training = datasample(tCali, 150000);
test = datasample(tCali, 50000);

% Discriminant Analysis
MdlLinear = fitcdiscr(training(:, 4:8), training(:, 9));
prediction = MdlLinear.predict(test);
figure;
confusionchart(test{:, 9}, prediction);
title("Confusion Matrix for Discriminant Analysis")

% KNN
MdlKNN = fitcknn(training(:, 4:8), training(:, 9));
predictionKNN = MdlKNN.predict(test);
figure;
confusionchart(test{:, 9}, predictionKNN);
title("Confusion Matrix for KNN");

% SVM Kernel
MdlKernel = fitckernel(training(:, 4:8), training(:, 9));
predictionKernel = MdlKernel.predict(test);
figure;
confusionchart(test{:, 9}, predictionKernel);
title("Confusion Matrix for SVM");

clear std;

%% PCA
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
% % Basic PCA process script 
% % this just transforms and plots data, it is up to you 
% % to provide the appropriate interpretation, which is the 
% % most valuable part of PCA 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  

% 3:end includes county, 4:end excludes county
tCaliPCA = tCali(:, 4:9);

[nrows, ncols] = size(tCaliPCA); 
X = zeros([nrows,ncols]); 
ss_i = zeros([1,ncols]);   % you could use this for the scree plot 
ss_cu = zeros([1,ncols]);  % you could use this for the cumulative scree plot 

tCaliPCA = table2array(tCaliPCA);
% tCaliPCA = rmmissing(tCaliPCA);

titles = ['date', 'time', 'county', 'alcohol involved', 'killed victims', 'injured victims', 'party count', 'highway status'];
means = mean(tCaliPCA);    
vars = var(tCaliPCA); 
stdevs = std(tCaliPCA); 

nfeatures = ncols;

% for i = 1:nfeatures
%     for j = 1:nfeatures
%         if i ~= j && i <= j
%             figure;
%             scatter(tCaliPCA(:, i), tCaliPCA(:, j), 'b');
%             [chart_title, ERRMSG] = sprintf('%s vs. %s', titles(i), titles(j));
%             xlabel(titles(i));
%             ylabel(titles(j));
%             title(chart_title);
%         end
%     end
% end

 
% 
% Mean center data 
% This is necessary so that everything is mean centered at 0 
% facilitates statistical and hypothesis analysis 
% 
for i=2:ncols 
    for j=1:nrows 
        X(j, 1) = tCaliPCA(j, 1);
        X(j,i) = -( means(:,i) - tCaliPCA(j,i)); 
    end 
end 
mean(X)  
 
% 
% Scale data 
% This is necessary so that all data has the same order, e.g.,  
% should not compare values in the thousands vs. values between 0 and 1 
% 
for i=2:ncols 
    for j=1:nrows 
        X(j, 1) = tCaliPCA(j, 1);
        X(j,i) = X(j,i) / stdevs(:,i);   
    end 
end         
var(X); 

[nrows, ncols] = size(X);

% 
% X is the original dataset 
% Ur will be the transformed dataset  
% S is covariance matrix (not normalized) 
% 
[U S V] = svd(X,0); 
Ur = U*S;


% Number of features to use 
f_to_use = nfeatures;      
feature_vector = 1:f_to_use; 

r = Ur;  % make a copy of Ur to preserve it,  we will randomize r  

muUr = mean(Ur);
covUr = cov(Ur);

% 
% Obtain the necessary information for Scree Plots 
% Obtain S^2 (and can also use to normalize S)   
% 
S2 = S^2; 
weights2 = zeros(nfeatures,1); 
sumS2 = sum(sum(S2)); 
weightsum2 = 0; 

for i=1:nfeatures 
    weights2(i) = S2(i,i)/sumS2; 
    weightsum2 = weightsum2 + weights2(i); 
    weight_c2(i) = weightsum2; 
end 


% Plotting Scree Plots 
figure; 
plot(weights2,'x:b'); 
grid; 
title('Scree Plot'); 

figure; 
plot(weight_c2,'x:r'); 
grid; 
title('Scree Plot Cumulative');

for i=1:nfeatures 
    for j=1:nfeatures 
        Vsquare(i,j) = V(i,j)^2; 
        if V(i,j)<0 
            Vsquare(i,j) = Vsquare(i,j)*-1; 
        else  
            Vsquare(i,j) = Vsquare(i,j)*1; 
        end 
    end 
end 


for i = 1:nfeatures
    figure; 
    bar(Vsquare(:,i),0.5); 
    grid; 
    ymin = min(Vsquare(:,1)) + (min(Vsquare(:,1))/10); 
    ymax = max(Vsquare(:,1)) + (max(Vsquare(:,1))/10); 
    axis([0 nfeatures ymin ymax]); 
    xlabel('Feature index'); 
    ylabel('Importance of feature'); 
    [chart_title, ERRMSG] = sprintf('Loading Vector %d',i); 
    title(chart_title); 
end

for i = 1:nfeatures
    for j = 1:nfeatures
        if i ~= j && i <= j
            figure;
            scatter(Ur(:, i), Ur(:, j), 'b');
            [chart_title, ERRMSG] = sprintf('PC %d vs. PC %d', i, j);
            xlabel(sprintf('PC %d', i));
            ylabel(sprintf('PC %d', j));
            title(chart_title);
        end
    end
end

close all;