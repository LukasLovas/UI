import random
import time
import math
import tkinter as tk
from Node import Node
from Solution import Solution_Annealing
from Solution import Solution_Tabu


def generate_random_location(width, height):
    x = random.randint(0, width)
    y = random.randint(0, height)
    return x, y


def calculate_total_distance_for_solution(solution):
    total_distance = 0
    num_cities = len(solution)

    for i in range(num_cities):
        from_city = solution[i]
        to_city = solution[(i + 1) % num_cities]
        total_distance += math.sqrt((from_city.x - to_city.x) ** 2 + (from_city.y - to_city.y) ** 2)

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


def get_distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def initialize_first_solution(nodes):
    current_node = random.choice(nodes)
    best_solution = [current_node]

    free_nodes = set(nodes)
    free_nodes.remove(current_node)
    while free_nodes:
        next_node = min(free_nodes, key=lambda x: get_distance(current_node, x))
        free_nodes.remove(next_node)
        best_solution.append(next_node)
        current_node = next_node
    print("Vzdialenosť prvotného riešenia: " + str(calculate_total_distance_for_solution(best_solution)))
    return best_solution


# def get_neighbors(solution, max_neighbours):
#     neighbors = []
#     while len(neighbors) <= max_neighbours:
#         neighbor = solution.copy()
#         pos1 = random.randint(0, 19)
#         pos2 = random.randint(0, 19)
#         if pos1 == pos2:
#             pass
#         else:
#             neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
#             if neighbor not in neighbors:
#                 neighbors.append(neighbor)
#
#     return neighbors

def get_neighbors(solution):
    neighbors = []
    num_cities = len(solution)

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            neighbor = solution[:i] + solution[i:j][::-1] + solution[j:]
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
    timer_start = time.time()
    current_solution = initialize_first_solution(nodes)
    best_solution = current_solution
    tabu_list = []
    iteration = 0

    while iteration <= max_iterations:
        neighbors = get_neighbors(current_solution)
        best_neighbor = fitness_function(neighbors, tabu_list)

        if compare_solutions(best_neighbor, best_solution):
            best_solution = best_neighbor

        current_solution = best_neighbor
        update_tabu_list(current_solution, tabu_list, tabu_list_length)

        iteration += 1
    timer_stop = time.time() - timer_start
    return Solution_Tabu(len(nodes), max_iterations, best_solution, timer_stop)


def update_tabu_list(current_solution, tabu_list, tabu_list_length):
    tabu_list.append(current_solution)

    if len(tabu_list) > tabu_list_length:
        tabu_list.pop(0)


# ------------------- END TABU ------------------------------- #

# -------------------- ANNEALING ------------------------------ #
def simulated_annealing(nodes, initial_temperature, cooling_rate, max_iterations):
    timer_start = time.time()
    current_solution = initialize_first_solution(nodes)
    best_solution = current_solution
    current_distance = calculate_total_distance_for_solution(current_solution)
    best_distance = current_distance
    iteration = 0
    best_solution_change = 0

    temperature = initial_temperature

    while iteration <= max_iterations:
        neighbor = random.choice(get_neighbors(current_solution))
        neighbor_distance = calculate_total_distance_for_solution(neighbor)
        distance_difference = neighbor_distance - current_distance
        try:
            acceptance_probability = math.exp(-distance_difference / temperature)
        except OverflowError:
            acceptance_probability = math.inf

        if distance_difference < 0 or random.random() < acceptance_probability:
            current_solution = neighbor
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_solution_change += 1
                best_solution = current_solution
                best_distance = current_distance

        temperature *= cooling_rate
        iteration += 1

    timer_stop = time.time() - timer_start
    return Solution_Annealing(len(nodes), initial_temperature, cooling_rate, max_iterations, best_solution_change,
                              best_solution,timer_stop)


# ----------------------------------- END ANNEALING ------------------------------ #
def start():
    result = None
    while result is None:
        mode = int(
            input("Vyber algoritmus: \n\t 1. Tabu search (input 1)\n\t 2. Simulated annealing (input 2)\n\n Input: "))
        if mode == 1:
            try:
                tabu_list_lenght = int(input("Zadaj velkosť tabu listu: "))
            except ValueError:
                tabu_list_lenght = 30
            max_iterations = 50
            result = tabu_search(nodes, tabu_list_lenght, max_iterations)
        elif mode == 2:
            max_iterations = 5000
            result = simulated_annealing(nodes, 10000, 0.85, max_iterations)
    return result, mode


def visualize_result(result):
    def start_animation():
        order_of_cities = []

        for city in result:
            city.circle = canvas.create_oval(city.x - 5, city.y - 5, city.x + 5, city.y + 5, fill="red", width=2)
            order_of_cities.append(city)

        first_city = order_of_cities[0]
        canvas.itemconfig(first_city.circle, fill="gold")
        for i in range(1, len(order_of_cities)):
            from_city = order_of_cities[i - 1]
            to_city = order_of_cities[i]

            canvas.itemconfig(to_city.circle, fill="green")

            canvas.create_line(from_city.x, from_city.y, to_city.x, to_city.y, fill="blue", arrow=tk.LAST)
            print(str(math.sqrt((from_city.x - to_city.x) ** 2 + (from_city.y - to_city.y) ** 2)))
            canvas.update()
            canvas.after(800)

            canvas.itemconfig(to_city.circle, fill="red")

        last_city = order_of_cities[-1]
        to_first_city = order_of_cities[0]
        canvas.create_line(last_city.x, last_city.y, to_first_city.x, to_first_city.y, fill="blue", arrow=tk.LAST)
        print(str(math.sqrt((last_city.x - to_first_city.x) ** 2 + (last_city.y - to_first_city.y) ** 2)))

        canvas.update()

    window = tk.Tk()
    window.title("Traveling Salesman Problem Vizualizácia")
    canvas = tk.Canvas(window, width=500, height=500)
    canvas.pack()

    start_button = tk.Button(window, text="Štart", command=start_animation)
    start_button.pack()

    window.mainloop()


if __name__ == '__main__':
    width = int(input("Urči vertikálnu vzdialenosť: "))
    height = int(input("Urči horizontálnu vzdialenosť: "))
    nodes = generate_random_node_locations(width, height)
    result, mode = start()
    if mode == 1:
        print("Vzdialenosť najlepšieho riešenia: " + str(calculate_total_distance_for_solution(result.solution)))
        print(f"Najlepšie riešenie malo: {result.num_of_cities} miest\n"
              f"pre {result.max_iterations} iterácií algoritmus potreboval {result.timer} sekúnd na zbehnutie programu")
    else:
        print("Vzdialenosť najlepšieho riešenia: " + str(calculate_total_distance_for_solution(result.solution)))
        print(f"Najlepšie riešenie malo: {result.num_of_cities} miest\n"
              f"začiatočnú teplotu: {result.initial_temp}\n"
              f"ochladzovanie: {result.cooling_rate}\n"
              f"pre {result.max_iterations} iterácií algoritmu sa najlepšie riešenie zmenilo {result.best_solution_change_counter} krát.\n"
              f"Algoritmus potreboval {result.timer} sekúnd na zbehnutie programu")
    visualize_result(result.solution)
