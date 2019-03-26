% 1.DESCRIPTIVE STATISTICS

% a) Generate the time series of continuous returns (r1,r2) associated with the "two assets" for the data history 14th Aug 2008 to 
%    17th Aug 2017.

% Variable generation
USD_I_7 = TimeSeriesI.USD_I_7;
USD_II_5 = TimeSeriesII.USD_II_5;
dates = (39682:7:42965).';
dates_1 = [39675; dates];

% Plot of values 
figure;
plot(dates_1,USD_I_7,'r',dates_1,USD_II_5,'b');
axis([ 39675 42965 min(min(USD_I_7),min(USD_II_5)) max(max(USD_I_7),max(USD_II_5)) ]);
datetick('x',12,'keeplimits');
title('Weekly Values from 15-Aug-08 to 18-Aug-17');
legend('USD-I-7','USD-II-5','location','northwest');
xlabel('Weeks');
ylabel('Weekly Values');

% Generation of continuous returns 
[r1] = tick2ret(USD_I_7, TimeSeriesI.Date, 'Continuous');
[r2] = tick2ret(USD_II_5, TimeSeriesII.Date,'Continuous');

% Plot of continuous returns
figure;
plot(dates,r1,'r',dates,r2,'b');
axis([ 39682 42965 min(min(r1),min(r2)) max(max(r1),max(r2)) ]);
bound_zero = refline(0,0);
bound_zero.Color = 'k'
datetick('x',12,'keeplimits');
title('Weekly Continuous Returns from 15-Aug-08 to 18-Aug-17');
legend('USD-I-7','USD-II-5','location','southwest');
xlabel('Weeks');
ylabel('Weekly Continuous Returns');

% Test normality of log returns

    % 1.96 Standard Deviations
    for i=1:470
    bound_up1(i,1) = mean(r1)+1.96*std(r1);
    bound_down1(i,1) = mean(r1)-1.96*std(r1);
    bound_up2(i,1) = mean(r2)+1.96*std(r2);
    bound_down2(i,1) = mean(r2)-1.96*std(r2);
    end
    figure;
    plot(dates,r1,'-r',dates,r2,'-b',dates,bound_up1,'--r',dates,bound_down1,'--r',dates,bound_up2,'--b',dates,bound_down2,'--b');
    axis([ 39682 42965 min(min(r1),min(r2)) max(max(r1),max(r2)) ]);
    bound_zero = refline(0,0);
    bound_zero.Color = 'k'
    datetick('x',12,'keeplimits');
    title('Weekly Continuous Returns from 15-Aug-08 to 18-Aug-17');
    legend('USD-I-7','USD-II-5','location','southwest');
    xlabel('Weeks');
    ylabel('Weekly Continuous Returns');
     
    % Histogram
    figure;
    histogram(r1);
    title('USD-I-7, Distribution of Weekly Continuous Returns from 15-Aug-08 to 18-Aug-17');
    xlabel('Weekly Continuous Returns');
    ylabel('Frequency');
    figure;
    histogram(r2);
    title('USD-II-5, Distribution of Weekly Continuous Returns from 15-Aug-08 to 18-Aug-17');
    xlabel('Weekly Continuous Returns');
    ylabel('Frequency');
    
    % Jarque-Bera test
    jb_r1 = jbtest(r1);
    jb_r2 = jbtest(r2);

% Histogram of lognormal prices
    figure;
    histogram(USD_I_7);
    title('USD-I-7, Distribution of Weekly Prices from 15-Aug-08 to 18-Aug-17');
    xlabel('Weekly Prices');
    ylabel('Frequency');
    figure;
    histogram(USD_II_5);
    title('USD-II-5, Distribution of Weekly Prices from 15-Aug-08 to 18-Aug-17');
    xlabel('Weekly Prices');
    ylabel('Frequency');

% b) Calculate the (expected) weekly returns (r1,r2) and the weekly standard deviations (o1,o2 ) for these two time series generated 
%    in step 1a) for the respective data history.   

% Returns of continuous returns
er_r1 = mean(r1);
er_r2 = mean(r2);

% Standard deviation of continuous returns
sd_r1 = std(r1);
sd_r2 = std(r2);

% c) Calculate and interpret the kurtosis (k1,k2) and the skewness (s1,s2) for the two time series (generated in step 1a).

% Kurtosis
kurtosis_r1 = kurtosis(r1);
kurtosis_r2 = kurtosis(r2);

% Skewness
skewness_r1 = skewness(r1);
skewness_r2 = skewness(r2);

% d) Compute the variance-covariance matrix (?) and their correlation matrix (rho12) for the two time series generated in step 1a). 
%    In addition compute the Spearman rank correlation for the two time series (generated in step 1a) and discuss the results.

% Variance-covariance matrix
cov_mat = cov(r1,r2);
  
% Correlation matrix  
corr_mat = corrcoef(r1,r2);
  
% Spearman correlation coefficient 
[RHO,PVAL] = corr(r1,r2,'type','Spearman')
  
% Scatter plot
figure;
scatter(r1,r2);
h = lsline;
set(h,'color','r')
title('Scatter Plot of Weekly Continuous Returns from 15-Aug-08 to 18-Aug-17');
xlabel('Returns of USD-I-7');
ylabel('Returns of USD-II-5');
  
% e) Apply the transformation f(x)=100*exp(x) to the two time series you have generated in step 1a). Compute the variance-covariance 
%    matrix (?) and their correlation matrix (rho12) for this two transformed time series. In addition compute the 
%    Spearman rank correlation for the trans-formed two time series and compare the results with those you have achieved in step 1d).
%    Discuss your observations.

% Transformation
r1_t = 100*exp(r1);
r2_t = 100*exp(r2);

% Plot of transformed returns
figure;
plot(dates,r1_t,'r',dates,r2_t,'b');
axis([ 39682 42965 min(min(r1_t),min(r2_t)) max(max(r1_t),max(r2_t)) ]);
bound_zero = refline(0,100);
bound_zero.Color = 'k'
datetick('x',12,'keeplimits');
title('Weekly Transformed Returns from 15-Aug-08 to 18-Aug-17');
legend('USD-I-7','USD-II-5','location','southwest');
xlabel('Weeks');
ylabel('Weekly Transformed Returns');
    
% Scatter plot
figure;
scatter(r1_t,r2_t);
h = lsline;
set(h,'color','r')
title('Scatter Plot of Weekly Continuous Returns from 15-Aug-08 to 18-Aug-17');
xlabel('Returns of USD-I-7');
ylabel('Returns of USD-II-5');

% Variance-covariance matrix
cov_mat_t = cov(r1_t,r2_t);

% Correlation matrix  
corr_mat_t = corrcoef(r1_t,r2_t);

% Spearman correlation coefficient 
[RHO_t,PVAL_t] = corr(r1_t,r2_t,'type','Spearman');

% ---------------------------------------------------------------------------------------------------------------------------------------------------

% 2.DESCRIPTIVE STATISTICS FOR QUARTILE RANGES

% a) Determine with respect to the first component the four quartile ranges of the two-dimensional return time series generated in 1a). Evaluate 
%    the variances and correlations for each quartile range.

% Computation of quartiles wrt r1
p = [0.25 0.5 0.75].';
quartile1 = quantile(r1, p);

% Partitioning of the time series into four quartile ranges
returns = [r1 r2];
r1_ordered = sortrows(returns,1);

quartile_range_r1_1 = r1_ordered(1:round(0.25*(470+1)),:);
quartile_range_r1_2 = r1_ordered(round(0.25*(470+1))+1: floor(0.5*(470+1)),:);
quartile_range_r1_3 = r1_ordered(floor(0.5*(470+1))+1:round(0.75*(470+1)),:);
quartile_range_r1_4 = r1_ordered(round(0.75*(470+1))+1:end,:);

% Plot of quartile ranges wrt r1
figure
plot(1:round(0.25*(470+1)),quartile_range_r1_1(:,1),'DisplayName','Q1 of r1','Color','r')
hold on
plot(1:round(0.25*(470+1)),quartile_range_r1_1(:,2),':','DisplayName','Q1 of r2','Color','r')
plot(round(0.25*(470+1))+1:floor(0.5*(470+1)),quartile_range_r1_2(:,1),'DisplayName','Q2 of r1','Color','g');
plot(round(0.25*(470+1))+1:floor(0.5*(470+1)),quartile_range_r1_2(:,2),':','DisplayName','Q2 of r2','Color','g');
plot(floor(0.5*(470+1))+1:round(0.75*(470+1)),quartile_range_r1_3(:,1),'DisplayName','Q3 of r1','Color','m');
plot(floor(0.5*(470+1))+1:round(0.75*(470+1)),quartile_range_r1_3(:,2),':','DisplayName','Q3 of r2','Color','m');
plot(round(0.75*(470+1))+1:470,quartile_range_r1_4(:,1),'DisplayName','Q4 of r1','Color','b');
plot(round(0.75*(470+1))+1:470,quartile_range_r1_4(:,2),':','DisplayName','Q4 of r2','Color','b');
hold off
grid on
axis([ 1 470 min(min(r1),min(r2)) max(max(r1),max(r2)) ]);
title('Four Quanrtile Ranges wrt USD-I-7');
ylabel('Ordered Weekly Log Returns');
legend('show','Location','Northwest');

% Var-cov matrix of the quartiles
cov_mat_qr_r1_1 = cov(quartile_range_r1_1(:,1),quartile_range_r1_1(:,2));
cov_mat_qr_r1_2 = cov(quartile_range_r1_2(:,1),quartile_range_r1_2(:,2));
cov_mat_qr_r1_3 = cov(quartile_range_r1_3(:,1),quartile_range_r1_3(:,2));
cov_mat_qr_r1_4 = cov(quartile_range_r1_4(:,1),quartile_range_r1_4(:,2));

% Correlations of the quartiles
quartile_range_r1_1_corr = corrcoef(quartile_range_r1_1(:,1),quartile_range_r1_1(:,2));
quartile_range_r1_2_corr = corrcoef(quartile_range_r1_2(:,1),quartile_range_r1_2(:,2));
quartile_range_r1_3_corr = corrcoef(quartile_range_r1_3(:,1),quartile_range_r1_3(:,2));
quartile_range_r1_4_corr = corrcoef(quartile_range_r1_4(:,1),quartile_range_r1_4(:,2));

quartile_range_r1_corr = [quartile_range_r1_1_corr quartile_range_r1_2_corr quartile_range_r1_3_corr quartile_range_r1_4_corr];

% b) Determine with respect to the second component the four quartile ranges of the two-dimensional return time series generated in 1a). Evaluate 
%    the variances and correlations for each quartile range.

% Computation of quartiles wrt r2
quartile2 = quantile(r2, p);

% Partitioning of the time series into four quartile ranges
r2_ordered = sortrows(returns,2);

quartile_range_r2_1 = r2_ordered(1:round(0.25*(470+1)),:);
quartile_range_r2_2 = r2_ordered(round(0.25*(470+1))+1: floor(0.5*(470+1)),:);
quartile_range_r2_3 = r2_ordered(floor(0.5*(470+1))+1:round(0.75*(470+1)),:);
quartile_range_r2_4 = r2_ordered(round(0.75*(470+1))+1:end,:);

% Plot of quartile ranges wrt r2
figure
plot(1:round(0.25*(470+1)),quartile_range_r2_1(:,2),'DisplayName','Q1 of r2','Color','r')
hold on
plot(1:round(0.25*(470+1)),quartile_range_r2_1(:,1),':','DisplayName','Q1 of r1','Color','r')
plot(round(0.25*(470+1))+1:floor(0.5*(470+1)),quartile_range_r2_2(:,2),'DisplayName','Q2 of r2','Color','g');
plot(round(0.25*(470+1))+1:floor(0.5*(470+1)),quartile_range_r2_2(:,1),':','DisplayName','Q2 of r1','Color','g');
plot(floor(0.5*(470+1))+1:round(0.75*(470+1)),quartile_range_r2_3(:,2),'DisplayName','Q3 of r2','Color','m');
plot(floor(0.5*(470+1))+1:round(0.75*(470+1)),quartile_range_r2_3(:,1),':','DisplayName','Q3 of r1','Color','m');
plot(round(0.75*(470+1))+1:470,quartile_range_r2_4(:,2),'DisplayName','Q4 of r2','Color','b');
plot(round(0.75*(470+1))+1:470,quartile_range_r2_4(:,1),':','DisplayName','Q4 of r1','Color','b');
hold off
grid on
axis([ 1 470 min(min(r1),min(r2)) max(max(r1),max(r2)) ]);
title('Four Quanrtile Ranges wrt USD-II-5');
ylabel('Ordered Weekly Log Returns');
legend('show','Location','Northwest')

% Var-cov matrix of the quartiles
cov_mat_qr_r2_1 = cov(quartile_range_r2_1(:,1),quartile_range_r2_1(:,2));
cov_mat_qr_r2_2 = cov(quartile_range_r2_2(:,1),quartile_range_r2_2(:,2));
cov_mat_qr_r2_3 = cov(quartile_range_r2_3(:,1),quartile_range_r2_3(:,2));
cov_mat_qr_r2_4 = cov(quartile_range_r2_4(:,1),quartile_range_r2_4(:,2));

% Correlations of the quartiles
quartile_range_r2_1_corr = corrcoef(quartile_range_r2_1(:,1),quartile_range_r2_1(:,2));
quartile_range_r2_2_corr = corrcoef(quartile_range_r2_2(:,1),quartile_range_r2_2(:,2));
quartile_range_r2_3_corr = corrcoef(quartile_range_r2_3(:,1),quartile_range_r2_3(:,2));
quartile_range_r2_4_corr = corrcoef(quartile_range_r2_4(:,1),quartile_range_r2_4(:,2));

quartile_range_r2_corr = [quartile_range_r2_1_corr quartile_range_r2_2_corr quartile_range_r2_3_corr quartile_range_r2_4_corr];

% ---------------------------------------------------------------------------------------------------------------------------------------------------

% 3.CHOLESKY DECOMPOSITION

% Apply the Cholesky decomposition to the covariance matrix (Sigma) and call the new matrix D. Re-member: Sigma=D D'.
D = chol(cov_mat,'lower');

% ---------------------------------------------------------------------------------------------------------------------------------------------------

% 4.RANDOM NUMBERS

% a) Generate 1040 pairs of random numbers X=[x1;x2] that are N(E,Sigma)-distributed, where E=[r1;r2] are the expected weekly returns evaluated  
%    in step 1b). Recall: X=E+D*Z. Plot these random numbers in a scatter diagram and interpret the point cloud.

% Generation of 1040 pairs of standard normally distributed random numbers
Z = normrnd(0,1,[1040,2]);
        
% Generation of 1040 pairs of random numbers distributed with the given vector of returns and the given var-cov matrix
corr_rnd = Z*D.';

for i=1:1040
X(i,1) = er_r1+corr_rnd(i,1);
X(i,2) = er_r2+corr_rnd(i,2);
end

% Scater Plot
figure;
scatter(X(:,1),X(:,2));
h = lsline;
set(h,'color','r')
title('Scatter Plot of 1040 pairs of random numbers X');
xlabel('Numbers generated from the expected return of USD-I-7');
ylabel('Numbers generated from the expected return of USD-II-5');

% Joint distribution
figure
hist3(X,[14 14]);
title('Distribution of 1040 pairs of random numbers X');
xlabel('USD-I-7'); ylabel('USD-II-5');

%---------------

% b) Calculate the moments - expected return,standard deviation, kurtosis and skewness ? as well as the variance-covariance matrix (?) and 
%    correla-tion matrix of the two simulated time series, each composed of 1040 random numbers. Compare them with the historical moments
%    of steps 1b) and 1c) and discuss.

% Expected return
e_X = mean(X,1);

% Standard-deviation
sd_X = std(X);

% Kurtosis
kutosis_X = kurtosis(X);

% Skewness
skewness_X = skewness(X);

% Variance-covariance matrix
cov_mat_X = cov(X);

% Correlation-Matrix
corr_mat_X = corrcoef(X); 

% c) Determine and interpret the weights W=[w_1;w_2] of the minimum variance portfolio. Estimate the expected weekly return of the minimum 
%    variance portfolio (rp) as well as its $95\%$ confidence interval with respect to the expected value and the return realization.

% Determination of the minimum variance portfolio
A = (cov_mat_X(1,1)+cov_mat_X(2,2)-2*corr_mat_X(2,1)*sd_X(1,1)*sd_X(1,2))/(cov_mat_X(1,1)*cov_mat_X(2,2)*(1-corr_mat_X(2,1)^2));
B = (cov_mat_X(1,1)*e_X(1,2)+cov_mat_X(2,2)*e_X(1,1)-corr_mat_X(2,1)*sd_X(1,1)*sd_X(1,2)*(e_X(1,1)+e_X(1,2)))/(cov_mat_X(1,1)*cov_mat_X(2,2)*(1-corr_mat_X(2,1)^2));
C = (cov_mat_X(1,1)*e_X(1,2)^2+cov_mat_X(2,2)*e_X(1,1)^2-2*corr_mat_X(2,1)*sd_X(1,1)*sd_X(1,2)*e_X(1,1)*e_X(1,2))/(cov_mat_X(1,1)*cov_mat_X(2,2)*(1-corr_mat_X(2,1)^2));
D = A*C-B^2;

er_mvp = B/A;
esd_mvp = sqrt(1/A);
evar_mvp  = esd_mvp^2;
weight_USD_I_7 = (cov_mat_X(2,2)-corr_mat_X(2,1)*sd_X(1,1)*sd_X(1,2))/(cov_mat_X(1,1)+cov_mat_X(2,2)-2*corr_mat_X(2,1)*sd_X(1,1)*sd_X(1,2));
weight_USD_II_5 = 1-weight_USD_I_7;

% -----------------------------------

% Draw the minimum variance frontier
weight_fund_1 = [-1:0.001:2].'
weight_fund_2 = [2:-0.001:-1].'

for i=1:3001
expected_return(i,1) = log(weight_fund_1(i,1)*exp(e_X(1,1))+weight_fund_2(i,1)*exp(e_X(1,2)));
variance(i,1) = weight_fund_1(i,1)^2*cov_mat_X(1,1)+weight_fund_2(i,1)^2*cov_mat_X(2,2)+2*weight_fund_1(i,1)*weight_fund_2(i,1)*cov_mat_X(2,1);
standard_deviation(i,1) = sqrt(weight_fund_1(i,1)^2*cov_mat_X(1,1)+weight_fund_2(i,1)^2*cov_mat_X(2,2)+2*weight_fund_1(i,1)*weight_fund_2(i,1)*cov_mat_X(2,1));
end

figure
plot(variance,expected_return);
hold on
plot(cov_mat_X(1,1),e_X(1,1),'co')
plot(cov_mat_X(2,2),e_X(1,2),'co')
plot(evar_mvp,er_mvp,'*')
grid on
title('Minimum Variance Frontier');
legend('Minimum Variance Frontier','USD-I-7','USD-II-5','Minimum Variance Portfolio','location','northwest');
xlabel('Portfolio Variance');
ylabel('Portfolio Return');
hold off

% Confidence interval for the expected value of the minimum variance portfolio
ci_er_mvp = [er_mvp-tinv(0.975,1039)*esd_mvp/sqrt(1040) er_mvp+tinv(0.975,1039)*esd_mvp/sqrt(1040)];

% Prediction interval for the return realization of the minimum variance portfolio
pi_er_mvp = [er_mvp-tinv(0.975,1039)*esd_mvp*sqrt(1+1/1040) er_mvp+tinv(0.975,1039)*esd_mvp*sqrt(1+1/1040)];


% ---------------------------------------------------------------------------------------------------------------------------------------------------

% 5) Simulation 

% Consider the 1040 pairs of random numbers as 20 paths, each path covering a period of 52 weeks. Evaluate, plot and interpret the profit 
% and loss distribution (P&L-distribution) of your minimum variance portfolio after 52 weeks ? you need not to  rebalance the portfolio when simulated. 
% (Set the initial asset prices to P1=100, P2=80 and apply the weights.)

% Creating an array of 20 double columns containing groups of 52 random number generated for the first and second asset
initial = 1
for i=1:20
    path(1:52,1:2,i)=[X(initial:i*52,1:2)]
    initial=i*52+1
end

% Cumulative Returns for the 20 series of the first random number for investing 100
for j=1:20
P_1 = 100
    for i=1:52
    cumul_1(i,j) = P_1*exp(path(i,1,j))
    P_1=cumul_1(i,j)
    end
end

% Cumulative Returns for the 20 series of the second random number for investing 80
for j=1:20
P_2 = 80
    for i=1:52
    cumul_2(i,j) = P_2*exp(path(i,2,j))
    P_2=cumul_2(i,j)
    end
end
  
% Putting the two above together: [100 cumulative ; 80 cumulative] repeated 20 times
  position_1 = 1
  position_2 = 2
  for i=1:20
      cumul_12(1:52,position_1)= cumul_1(1:52,i)
      position_1 = position_1+2
      cumul_12(1:52,position_2)= cumul_2(1:52,i)
      position_2 = position_2+2
  end
  
 % Final value of the portfolio with minimum variance weights 
 position_1 = 1
 position_2 = 2
 for i=1:20
 port_final_value(1,i) = weight_USD_I_7*cumul_12(52,position_1)+weight_USD_II_5*cumul_12(52,position_2)
 position_1 = position_1+2
 position_2 = position_2+2
 end
 
 % Initial value of the portfolio with minimum variance weights
 port_initial_value = weight_USD_I_7*100+weight_USD_II_5*80
 
 % Profit and loss of minimum variance portfolio
 p_and_loss = (port_final_value-port_initial_value).'
 p_and_loss_ordered = sortrows(p_and_loss,1);

% Profit and loss of individual assets
for i=1:20
    p_and_loss_a1(1,i) = cumul_1(52,i)-100;
    p_and_loss_a2(1,i) = cumul_2(52,i)-80;
end   
p_and_loss_a1 = p_and_loss_a1.'
p_and_loss_a2 = p_and_loss_a2.'
p_and_loss_a1_ordered = sortrows(p_and_loss_a1,1);
p_and_loss_a2_ordered = sortrows(p_and_loss_a2,1);

 % Plot profit and loss
bins=[0.05:0.05:1];
for i=1:20
constant(1,i) = 0
end
figure
plot(bins,p_and_loss_ordered,'b-*');
hold on
plot(bins,p_and_loss_a1_ordered,'r-');
plot(bins,p_and_loss_a2_ordered,'m-');
plot(bins,constant,'c--');
grid on
title('P&L Simulation');
xlabel('Probability');
ylabel('Absolute Profit/Loss');
hold off


figure
plot(p_and_loss_ordered,bins,'b-*');
hold on
plot(p_and_loss_a1_ordered,bins,'r-');
plot(p_and_loss_a2_ordered,bins,'m-');
grid on
title('P&L Simulation');
xlabel('Probability');
ylabel('Absolute Profit/Loss');
hold off

% Shortfall risk
for i=1:20
    total_return(i,1) = log(port_final_value(1,i)/port_initial_value)
    total_return_a1(i,1) = log(cumul_1(52,i)/100);
    total_return_a2(i,1) = log(cumul_2(52,i)/80);
end 

total_return_ordered = sortrows(total_return,1);
total_return_a1_ordered = sortrows(total_return_a1,1);
total_return_a2_ordered = sortrows(total_return_a2,1);

figure
plot(bins,total_return_ordered,'b-');
hold on
plot(bins,total_return_a1_ordered,'r-');
plot(bins,total_return_a2_ordered,'m-');
grid on
title('P&L Simulation');
xlabel('Shortfall Probability');
ylabel('Total Return');
hold off


