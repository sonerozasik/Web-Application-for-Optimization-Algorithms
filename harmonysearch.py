import random
import matplotlib.pyplot as plt

def harmony_search(cost_function, harmony_memory_size, harmony_memory_consideration_rate, pitch_adjustment_rate, max_iterations):
    # Initialize the harmony memory with random solutions
    harmony_memory = [cost_function() for _ in range(harmony_memory_size)]
    costs = []  # Track the cost of each solution
    
    for i in range(max_iterations):
        # Generate a new solution by adjusting the pitch of a random solution in the harmony memory
        new_solution = adjust_pitch(random.choice(harmony_memory), pitch_adjustment_rate)
        new_cost = cost_function(new_solution)  # Evaluate the cost of the new solution
        costs.append(new_cost)
        
        # Consider adding the new solution to the harmony memory
        if random.random() < harmony_memory_consideration_rate:
            # If the harmony memory is full, replace the worst solution
            if len(harmony_memory) == harmony_memory_size:
                worst_solution = min(harmony_memory, key=cost_function)
                harmony_memory.remove(worst_solution)
            
            # Add the new solution to the harmony memory
            harmony_memory.append(new_solution)
    
    # Show a graph of the cost of the solutions over time
    plt.plot(costs)
    plt.show()
    
    # Return the best solution in the harmony memory
    return min(harmony_memory, key=cost_function)

def adjust_pitch(solution, pitch_adjustment_rate):
    # Generate a new solution by randomly perturbing the values of the solution
    new_solution = []
    for value in solution:
        if random.random() < pitch_adjustment_rate:
            # Adjust the value by a random amount
            value += random.uniform(-1, 1)
        new_solution.append(value)
    return new_solution
