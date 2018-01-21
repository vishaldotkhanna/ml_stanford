function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

m = size(X, 1);
n = size(theta);
h_linear = sum(X .* theta', 2);
h_logistic = sigmoid(h_linear);
cost_i = (y .* log(h_logistic)) + (1 - y) .* log(1 - h_logistic);
reg_term1 = sum((theta .^ 2)(2:n));  %Not including theta(1).
J = (-sum(cost_i) / m) + (lambda * reg_term1) / (2 * m);

grad = sum(X' * (h_logistic - y), 2) / m + (lambda * [0; theta(2:n)]) / m;




% =============================================================

end
