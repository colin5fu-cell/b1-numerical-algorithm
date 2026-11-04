clear;
close all;
clc;

% Parameters
omega = 2.0;
phi = pi/6;

% Start and end times
x_start = 0.0;
x_end = 2*pi;

%Initiate arrays
numSteps_values = [32 64 128 256 512]';
h_values = (x_end-x_start)./numSteps_values;
difference_euler = zeros(size(h_values));
difference_rk4 = zeros(size(h_values));

%Iterate through to compute euler and rk4 at different steps
for j=1:numel(h_values)
   
    
    numSteps = numSteps_values(j);
    h = h_values(j);

    x = linspace(x_start,x_end,numSteps+1)';
    fn = sin(omega*x + phi);

    % Size vectors for Euler and PC methods
    y = zeros(size(x));
    dydx = zeros(size(x));
    
    y_rk4 = zeros(size(x));
    dydx_rk4 = zeros(size(x));

    % Initial values
    y(1) = sin(phi);
    dydx(1) = omega*cos(phi);

    y_rk4(1) = sin(phi);
    dydx_rk4(1) = omega*cos(phi);

    % Perform Euler
    for i=1:(length(x)-1)
        y(i+1)=y(i)+h*dydx(i);
        dydx(i+1)=dydx(i)-h*omega^2*y(i);
    end
    
    % Perform Runge-Kutta
    for i = 1:(length(x)-1)

        k1_y = h*dydx_rk4(i);
        k1_v = -h*omega^2*y_rk4(i);

        k2_y = h*(dydx_rk4(i) + k1_v/2);
        k2_v = -h*omega^2*(y_rk4(i) + k1_y/2);

        k3_y = h*(dydx_rk4(i) + k2_v/2);
        k3_v = -h*omega^2*(y_rk4(i) + k2_y/2);

        k4_y = h*(dydx_rk4(i) + k3_v);
        k4_v = -h*omega^2*(y_rk4(i) + k3_y);

        y_rk4(i+1) = y_rk4(i) + (k1_y + 2*k2_y + 2*k3_y + k4_y)/6;

        dydx_rk4(i+1) = dydx_rk4(i) + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6;
    end

    difference_euler(j) = max(abs(fn-y));

    difference_rk4(j) = max(abs(fn-y_rk4));
end


%% Display results
results_C = table( ...
    numSteps_values,h_values, ...
    difference_euler,difference_rk4, ...
    'VariableNames', ...
    {'Number_of_steps','h', ...
     'Euler_max_error','RK4_max_error',});

disp(results_C);

%% Error versus step size
figure;

loglog(h_values,difference_euler,'-o', ...
       h_values,difference_rk4,'-s', ...
       'LineWidth',1.5);

legend('Euler','RK4','Location','best');

xlabel('Step size h');
ylabel('Maximum absolute error');
title('Global error versus step size');

grid on;