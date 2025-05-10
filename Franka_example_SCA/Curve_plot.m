clc
clear
close all

load('Tau_SCA_active.mat');
Tau_SCA_active = a;
load('Tau_SCA_inactive.mat');
Tau_SCA_inactive = a;

load('mindist_active.mat');
mindist_active = a;
load('mindist_inactive.mat');
mindist_inactive = a;

stepsize = 0.001;
duration = 10;

figure(1)
subplot(2, 1, 1)
plot(0:stepsize:(duration - stepsize), Tau_SCA_active, 'LineWidth', 1.5, 'Color', 'black');
hold on
plot(0:stepsize:(duration - stepsize), Tau_SCA_inactive, 'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410]);
hold on
plot(0:stepsize:(duration - stepsize), zeros(1, duration/stepsize), 'Color', 'r');
hold on
plot(0:stepsize:(duration - stepsize), 10*ones(1, duration/stepsize), 'Color', 'g', 'LineStyle','--');

xlabel('Time (s)', 'FontName', 'Times New Roman');
ylabel('$h_{SCA}(q(t))$', 'Interpreter','latex');
legend('SCA constraint active', 'SCA constraint inactive', 'FontName', 'Times New Roman', 'FontSize', 10);
grid on
text(0, 0, 'SCA boundary', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'red');
text(0, 10, 'Minimal threshold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'red');

title('Evolution of $h_{SCA}(q(t))$', 'Interpreter','latex');
hold off

subplot(2, 1, 2)
plot(0:stepsize:(duration - stepsize), mindist_active, 'LineWidth', 1.5, 'Color', 'black');
hold on
plot(0:stepsize:(duration - stepsize), mindist_inactive, 'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410]);
hold on
plot(0:stepsize:(duration - stepsize), zeros(1, duration/stepsize), 'Color', 'r');
hold on

xlabel('Time (s)', 'FontName', 'Times New Roman');
ylabel('Minimal distance', 'FontName','Times New Roman');
ylim([-0.05, 0.55]);
legend('SCA constraint active', 'SCA constraint inactive', 'FontName', 'Times New Roman', 'FontSize', 10);
grid on
text(0, 0, 'SCA boundary', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'red');
title('Minimal distance between link1 and link6', 'FontName', 'Times New Roman');
hold off




