function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
    i = 1;
    for iter = 1:num_iters

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %
        
        % We use tmp so we can acheive simultaneous calculation
        h = ((X * theta) - y);
        thetaChange_1 = alpha*(1/m) * sum(h .* X(:, 1))
        tmp1 = theta(1) - thetaChange_1;
        
        thetaChange_2 = alpha*(1/m) * sum(h .* X(:, 2));
        tmp2 = theta(2) - thetaChange_2;
        
        theta(1) = tmp1;
        theta(2) = tmp2;

        % ============================================================

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);

    end

end
