function pltError(error_history)
    % Function to plot the error convergence over iterations
    % Input:
    % error_history - Vector containing the error norm at each iteration

    % Determine the number of iterations
    iterations = 1:length(error_history);

    % Create the plot
    figure;
    plot(iterations, error_history, '-o');
    xlabel('Iteration');
    ylabel('Error Norm');
    title('Error Convergence in Gradient Method');
    grid on;
end
