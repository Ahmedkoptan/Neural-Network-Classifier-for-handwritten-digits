function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%adding 1 to the dataset
X=[ones(size(X,1),1) X];
%initialize Delta variables to 0
Delta2=0;Delta1=0;

%looping over all examples
for t=1:m
    %encoding y
    Y=zeros(m,num_labels);
    if(y(t)==0)
        Y(t,num_labels)=1;
    else
        Y(t,y(t))=1;
    end
    %forward prop
    A_1=X(t,:)';
    Z_2=Theta1*A_1;
    A_2=[1;sigmoid(Z_2)];
    Z_3=Theta2*A_2;
    h=sigmoid(Z_3);
    %computing cost
    J=J+((1/m)*[-Y(t,:) -(1-Y(t,:))]*[log(h);log(1-h)]);
    
    %back prop
    delta3=h-(Y(t,:)');
    delta2=(Theta2')*delta3.*[1;sigmoidGradient(Z_2)];
    delta2=delta2(2:end);
    Delta2=Delta2+delta3*(A_2');
    Delta1=Delta1+delta2*(A_1');
end
%This part is not affected by back prop code added previously
%adding regularization term
t1=Theta1(:,2:end);
t1=t1(:);
t2=Theta2(:,2:end);
t2=t2(:);
J=J+((lambda/(2*m))*((t1'*t1)+(t2'*t2)));

%this is is affected by back prop
Theta1_grad=(1/m)*Delta1;
Theta2_grad=(1/m)*Delta2;

%adding regularization after back prop
d1=[zeros(size(Delta1,1),1) ones(size(Delta1,1),size(Delta1,2)-1)]*(lambda/m);
d2=[zeros(size(Delta2,1),1) ones(size(Delta2,1),size(Delta2,2)-1)]*(lambda/m);
Theta1_grad=Theta1_grad+(d1.*Theta1);
Theta2_grad=Theta2_grad+(d2.*Theta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
