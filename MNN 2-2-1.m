%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Author: Mohammed Kashwah %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc


%initialization step
eta     = 0.1;              %learning rate
theta   = 0.001;            %threshold
j_w     = [1];              %arbitrary value
delta_jw = [1];
epoch   = 1;                %number of epochs
x_1     = [-1 -1 1 1];      %input 1
x_2     = [-1 1 -1 1];      %input 2
t       = [-1 1 1 -1];      %target output
bias    = 1;                %bias
input_augmented = [ones(1, length(t)); x_1; x_2];       %ones(1, length(t)) is for bias
input_transposed = input_augmented';                    %transpose the input for ease of calculations


%initialize w_ji =/= 0
w_ji = [1 1; 1 1; 1 1];         %w_ji = [w_1_bias       w_2_bias
                                %        w11            w21 
                                %        w12            w22]
                                       



%initialize w_kj =/= 0
w_kj = [1; 1; 1];                 %w_kj = [w_3_bias
                                %           w3_1
                                %           w3_2    ]

                         
while delta_jw > theta
    epoch = epoch+1;
    
    net_j = input_transposed * w_ji;        %net activation
    y_j   = tanh(net_j);
    y_j_augmented = [ones(length(x_1),1), y_j];     %including the bias
    
    net_k = y_j_augmented * w_kj;           %net activation for output

    z_k = tanh(net_k);
    %calculate j(w)
    t_z_difference = t - z_k';
    j_w(epoch) = 0.5* sum(t_z_difference.^2);        %0.5 * ||t - z_k||^2
%     j_w(epoch) = 0.5* (norm(t_z_difference))^2;        %0.5 * ||t - z_k||^2

    
    %backpropagation

    %between output and hidden layer
    sig_d_k = 1- (tanh(net_k)).^2;      %f'(net_k)
    
%     delta_k = (t - z_k') * sig_d_k;
    delta_k = sum((t - z_k')) * sig_d_k;                 %sensitivity delta_k

    delta_w_kj = eta * delta_k' * y_j_augmented;     %[delta_w_bias_3    delta_w3_1 delta_w3_2]

%     w_kj = w_kj + delta_w_kj';                      %w_kj(m+1)

    %between the hidden layer and the input
    sig_d_j = 1- (tanh(net_j)).^2;      %f'(net_j)
%     delta_j = sig_d_j' .* sum(w_kj * delta_k);
    delta_j = sum((t - z_k')* sig_d_k * w_kj) * sig_d_j;    %sensitivity delta_j

    delta_w_ji = eta * delta_j' * input_transposed;

    w_ji = w_ji + delta_w_ji';      %update w_ji
    w_kj = w_kj + delta_w_kj';      %update w_kj
    
    delta_jw = abs(j_w(epoch) - j_w(epoch - 1));        %find delta j(w)
    
end

figure
plot(1:1:epoch, j_w)
title('Cost Function J(w) vs number of epochs')
xlabel('# of epochs')
ylabel('cost function J(w)')
grid on



fprintf("number of epochs needed for convergance = " + epoch + "\n")

