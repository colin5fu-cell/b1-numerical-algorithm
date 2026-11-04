clear; close all; clc

%% Define variables, Generate original and noisy data
rng(1);

T = 2*pi;
dt = 0.001;
omega = 2;
phi = pi/6;
sigma = 0.01;

t = (0:dt:T)';
original_data = sin(omega*t + phi);
noisy_data = original_data + sigma*randn(size(t));

%% Define k and h values, initiate the arrays used to store errors in forward, backward and central differentiation
k_values = [1 2 4 8 16 32 64 128 256 512 1024];
h = k_values*dt;

difference_forward = zeros(size(h));
difference_backward = zeros(size(h));
difference_central = zeros(size(h));

%% Calculate derivatives and errors for each h
for j = 1:numel(h)

    k = k_values(j);
    
    % Subsample the data according to the current step size
    t_sub = t(1:k:end);
    y_sub = noisy_data(1:k:end);

    % All three methods estimate derivatives at these interior points
    t_inner = t_sub(2:end-1);

    exact_derivative = omega*cos(omega*t_inner + phi);

    forward_derivative = (y_sub(3:end) - y_sub(2:end-1))/h(j);

    backward_derivative = (y_sub(2:end-1) - y_sub(1:end-2))/h(j);

    central_derivative = (y_sub(3:end) - y_sub(1:end-2))/(2*h(j));

    % Calculate L2 error
    difference_forward(j) = sqrt(h(j) * sum((forward_derivative - exact_derivative).^2));

    difference_backward(j) = sqrt(h(j) * sum((backward_derivative - exact_derivative).^2));

    difference_central(j) = sqrt(h(j) * sum((central_derivative - exact_derivative).^2));
end

%% Error versus step size
figure
loglog(h,difference_forward,'-o', h,difference_backward,'-o', h,difference_central,'-o')

legend('forward','backward','central')
xlabel('h')
ylabel('L2 error')
title('Numerical differentiation under noise')
grid on

%% Original versus noisy data
figure
plot(t,noisy_data,'.','Color',[0.65 0.65 0.65])
hold on
plot(t,original_data,'k-','LineWidth',1.5)

legend('noisy data','original data')
xlabel('t')
ylabel('y(t)')
title('Original versus noisy data')
grid on

%% Display parameters and errors
fprintf('omega = %.2f, phi = %.2f, sigma = %.2f, dt = %.2f\n', omega,phi,sigma,dt);

results = table(h',difference_forward', difference_backward',difference_central','VariableNames',{'h','Forward_Error','Backward_Error','Central_Error'});

disp(results)