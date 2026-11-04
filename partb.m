clearvars; close all; clc

%% Define variables, calculate real and noisy y data, same as part A
rng(1);

T = 2*pi;
dt = 0.001;
omega = 2.0;
phi = pi/6;
sigma = 0.01;

t = (0:dt:T)';

y_true = sin(omega*t + phi);

noise = sigma*randn(size(t));
y_noisy = y_true + noise;

y_derivative = omega*cos(omega*t + phi);

%% Centre and scale time
mu = mean(t);
s = std(t);
x = (t - mu)/s;

t_inner = t(2:end-1);
x_inner = x(2:end-1);

%% Calculate Base-grid central difference
central_derivative = (y_noisy(3:end) - y_noisy(1:end-2))/(2*dt);
exact_derivative = y_derivative(2:end-1);
difference_central = sqrt(dt * sum((exact_derivative - central_derivative).^2));

%% Initiate arrays
degrees = (1:10)';

condition_number = zeros(size(degrees));
difference_polynomial = zeros(size(degrees));
coefficients = cell(numel(degrees),1);

fitted_values = zeros(numel(t),numel(degrees));
polynomial_derivatives = zeros(numel(t_inner),numel(degrees));

%% Polynomial fitting, conditioning and differentiation
for j = 1:numel(degrees)

    degree = degrees(j);

    p = polyfit(x,y_noisy,degree);
    coefficients{j} = p;

    fitted_values(:,j) = polyval(p,x);

    V = zeros(numel(x),degree+1);

    for power = 0:degree
        V(:,power+1) = x.^power;
    end

    condition_number(j) = cond(V'*V);

    dp = polyder(p);

    polynomial_derivative = polyval(dp,x_inner)/s;

    polynomial_derivatives(:,j) = polynomial_derivative;

    difference_polynomial(j) = sqrt(dt * sum((exact_derivative - polynomial_derivative).^2));
end

results_B1 = table(degrees,condition_number, 'VariableNames',{'Degree','Condition_number'});

central_errors = repmat(difference_central,numel(degrees),1);

results_B2 = table(degrees,difference_polynomial,central_errors, 'VariableNames', {'Degree','Polynomial_Error','Central_Error'});

disp(results_B1);
disp(results_B2);

%% Figure 1: condition number versus degree
figure

semilogy(degrees,condition_number,'-o','LineWidth',1.5)

xlabel('Polynomial degree')
ylabel('cond(V^T V)')
title('Condition number versus polynomial degree')

xticks(degrees)
grid on

%% Figure 2: derivative error comparison
figure

semilogy(degrees,difference_polynomial,'-o','LineWidth',1.5)
hold on

semilogy(degrees,central_errors,'--','LineWidth',1.5)

legend('Polynomial derivative', 'Central difference (h = dt)', 'Location','best')

xlabel('Polynomial degree')
ylabel('Normalized discrete L2 error')
title('Derivative error comparison')

xticks(degrees)
grid on

%% Figure 3: demo of degree 9

degree_to_plot = 9;

plot_index = find(degrees == degree_to_plot);

figure

% Original data, noisy data and polynomial fit
subplot(2,1,1)

plot(t,y_noisy,'.', 'Color',[0.65 0.65 0.65],'MarkerSize',3)
hold on

plot(t,y_true,'k-','LineWidth',1.5)
plot(t,fitted_values(:,plot_index),'r--','LineWidth',1.5)

legend('Noisy data','True signal','Polynomial fit', 'Location','best')

xlabel('Time t')
ylabel('y(t)')
title(sprintf('Polynomial fit: degree %d',degree_to_plot))
grid on

% Exact derivative and polynomial derivative
subplot(2,1,2)

plot(t_inner,exact_derivative,'k-','LineWidth',1.5)
hold on

plot(t_inner,polynomial_derivatives(:,plot_index), 'r--','LineWidth',1.5)

legend('Exact derivative','Polynomial derivative', 'Location','best')

xlabel('Time t')
ylabel('dy/dt')
title('Exact versus polynomial derivative')
grid on
