class Solution_Annealing:

    def __init__(self, num_of_cities, initial_temp, cooling_rate, max_iterations, best_solution_change_counter,
                 solution, timer):
        self.num_of_cities = num_of_cities
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.best_solution_change_counter = best_solution_change_counter
        self.solution = solution
        self.timer = timer


class Solution_Tabu:

    def __init__(self, num_of_cities, max_iterations, tabu_list_uses,
                 solution, timer):
        self.num_of_cities = num_of_cities
        self.max_iterations = max_iterations
        self.tabu_list_uses = tabu_list_uses
        self.solution = solution
        self.timer = timer
