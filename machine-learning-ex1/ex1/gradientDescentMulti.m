function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %sigma = [];
    %for j = 1:size(X, 2)
    %  sum_cost = 0.0;
    %  for i = 1:m
    %    sum_cost = sum_cost + (sum(theta' .* X(i, :)) - y(i)) * X(i, j);
    %   end;
    %  sigma = [sigma; sum_cost / m];
    %end;

    %theta = theta - alpha * sigma;

    theta = theta - (alpha / m) * (X' * (sum(X .* theta', 2) - y));
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
