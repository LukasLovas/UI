import random
import math
import tkinter as tk
from Node import Node


def generate_random_location(width, height):
    x = random.randint(0, width)
    y = random.randint(0, height)
    return x, y


def calculate_total_distance_for_solution(solution):
    total_distance = 0
    num_cities = len(solution)

    for i in range(num_cities):
        from_city = solution[i]
        to_city = solution[(i + 1) % num_cities]  # Wrap around to the first city
        total_distance += math.sqrt((from_city.x - to_city.x) ** 2 + (from_city.y - from_city.y) ** 2)

    return total_distance


def generate_random_node_locations(width, height):
    n = int(input("Urči počet miest: "))
    used = []
    nodes = []
    for i in range(n):
        x, y = generate_random_location(width, height)
        if (x, y) not in used:
            used.append((x, y))
            nodes.append(Node(x, y))
    return nodes


def initialize_first_solution(nodes):
    initial_solution = random.sample(nodes, len(nodes))
    return initial_solution


def get_neighbors(solution, max_neighbours):
    neighbors = []
    while len(neighbors) <= max_neighbours:
        neighbor = solution.copy()
        pos1 = random.randint(0, 19)
        pos2 = random.randint(0, 19)
        if pos1 == pos2:
            pass
        else:
            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
            if neighbor not in neighbors:
                neighbors.append(neighbor)

    return neighbors


def compare_solutions(best_neighbor, best_solution):
    best_neighbor_distance = calculate_total_distance_for_solution(best_neighbor)
    best_solution_distance = calculate_total_distance_for_solution(best_solution)
    return best_neighbor_distance < best_solution_distance


def fitness_function(neighbors, tabu_list):
    best_neighbor = neighbors[0]
    best_distance = calculate_total_distance_for_solution(neighbors[0])

    for neighbor in neighbors:
        neighbor_distance = calculate_total_distance_for_solution(neighbor)

        if neighbor not in tabu_list and neighbor_distance < best_distance:
            best_neighbor = neighbor
            best_distance = neighbor_distance

    return best_neighbor


# -------------------------- TABU --------------------------------#
def tabu_search(nodes, tabu_list_length, max_iterations):
    current_solution = initialize_first_solution(nodes)
    best_solution = current_solution
    tabu_list = []
    iteration = 0
    max_neighbours = 150

    while iteration < max_iterations:
        neighbors = get_neighbors(current_solution, max_neighbours)
        best_neighbor = fitness_function(neighbors, tabu_list)

        if compare_solutions(best_neighbor, best_solution):
            best_solution = best_neighbor

        current_solution = best_neighbor
        update_tabu_list(current_solution, tabu_list, tabu_list_length)

        iteration += 1

    return best_solution


def update_tabu_list(current_solution, tabu_list, tabu_list_length):
    tabu_list.append(current_solution)

    if len(tabu_list) > tabu_list_length:
        tabu_list.pop(0)


# ------------------- END TABU -------------------------------#

# -------------------- ANNEALING ------------------------------#
def simulated_annealing():
    pass


# ----------------------------------- END ANNEALING ------------------------------ #
def start():
    mode = None
    result = None
    while result is None:
        mode = int(
            input("Vyber algoritmus: \n\t 1. Tabu search (input 1)\n\t 2. Simulated annealing (input 2)\n\n Input: "))
        if mode == 1:
            try:
                tabu_list_lenght = int(input("Zadaj velkosť tabu listu: "))
            except ValueError:
                tabu_list_lenght = 30
            result = tabu_search(nodes, tabu_list_lenght, max_iterations)
        elif mode == 2:
            result = simulated_annealing()
    return result


def visualize_result(result):
    def start_animation():
        # Initialize the order of cities visited
        order_of_cities = []

        # Draw cities in red (initial state)
        for city in result:
            city.circle = canvas.create_oval(city.x - 5, city.y - 5, city.x + 5, city.y + 5, fill="red", width=2)
            order_of_cities.append(city)

        # Color the first city gold
        first_city = order_of_cities[0]
        canvas.itemconfig(first_city.circle, fill="gold")

        # Traverse the cities
        for i in range(1, len(order_of_cities)):
            from_city = order_of_cities[i - 1]
            to_city = order_of_cities[i]

            # Paint the next city in green
            canvas.itemconfig(to_city.circle, fill="green")

            # Draw an arrow to the next city
            canvas.create_line(from_city.x, from_city.y, to_city.x, to_city.y, fill="blue", arrow=tk.LAST)
            canvas.update()  # Update the canvas to show the changes
            canvas.after(1000)  # Add a delay of 1000 milliseconds (1 second)

            # Paint the current city in red (leaving)
            canvas.itemconfig(to_city.circle, fill="red")

        # Connect the last city back to the starting city
        last_city = order_of_cities[-1]
        to_first_city = order_of_cities[0]
        canvas.create_line(last_city.x, last_city.y, to_first_city.x, to_first_city.y, fill="blue", arrow=tk.LAST)
        canvas.update()  # Update the canvas to show the final connection

    window = tk.Tk()
    window.title("Traveling Salesman Problem Visualization")
    canvas = tk.Canvas(window, width=500, height=500)
    canvas.pack()

    start_button = tk.Button(window, text="Start Animation", command=start_animation)
    start_button.pack()

    window.mainloop()


if __name__ == '__main__':
    width = int(input("Urči vertikálnu vzdialenosť: "))
    height = int(input("Urči horizontálnu vzdialenosť: "))
    nodes = generate_random_node_locations(width, height)
    max_iterations = 50
    result = start()
    print("Vzdialenosť najlepšieho riešenia: " + str(calculate_total_distance_for_solution(result)))
    visualize_result(result)
