import random as random
from Node import Node
import math


def generate_random_location(width, height):
    x = random.randint(0, width)
    y = random.randint(0, height)
    return x, y


def calculate_distance(nodes, current_node):
    distances = []
    for node in nodes:
        if current_node == node:
            distance = 0  # TODO poznamka: pri vybere najblizsieho mesta treba davat pozor na to, aby podmienka bola > 0, presne kvoli tomuto if-u
            distances.append(distance)
        else:
            x = abs(node.x - current_node.x)
            y = abs(node.y - current_node.y)
            distance = math.hypot(x, y)
            distances.append(distance)
    return distances


def generate_random_node_locations(width, height):
    n = int(input("Urči počet miest: "))
    used = []
    nodes = []
    for i in range(n):
        x, y = generate_random_location(width, height)
        if (x, y) not in used:
            used.append((x, y))
            nodes.append(Node(x, y, i))
    return nodes


def choose_starting_node(nodes):
    node = None
    for i in range(len(nodes)):
        print(f"No. {i + 1}: (x: {nodes[i].x} y: {nodes[i].y})")
    while node is None:
        start = int(input("Vyber poradové číslo mesta, z ktorého začať: ")) - 1
        if 0 <= start < len(nodes):
            node = nodes[start]
            node.order = 1
    return node

def simulated_annealing():
    pass

def initialize_first_solution(nodes):
    initial_solution = random.sample(nodes, len(nodes))
    return initial_solution


def get_neighbors(solution):
    neighbors = []
    while len(neighbors) <= 150:
        neighbor = solution.copy()
        pos1 = random.randint(0, 19)
        pos2 = random.randint(0, 19)
        if pos1 == pos2:
            pass
        else:
            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
            neighbor_order = [node.order for node in neighbor]
            if neighbor not in neighbors:
                neighbors.append(neighbor_order)

    return neighbors

def fitness_function(neighbors,tabu_list)
    best_neighbor = None
    best_distance = float('inf')  # Initialize to positive infinity

    for neighbor in neighbors:
        # Calculate the total distance (or cost) of the neighbor solution
        neighbor_distance = calculate_distance(neighbor)

        # Check if the neighbor solution is not in the tabu list
        if neighbor not in tabu_list and neighbor_distance < best_distance:
            best_neighbor = neighbor
            best_distance = neighbor_distance

    return best_neighbor

def tabu_search(nodes, tabu_list_length, max_iterations):
    current_solution = initialize_first_solution(nodes)  # Inicializácia prvého riešenia
    best_solution = current_solution
    tabu_list = []
    iteration = 0
    max_neighbours = 150

    while iteration < max_iterations:
        neighbors = get_neighbors(current_solution)
        best_neighbor = select_best_neighbor(neighbors, tabu_list)

        if is_better(best_neighbor, best_solution):
            best_solution = best_neighbor

        current_solution = best_neighbor
        update_tabu_list(tabu_list, current_solution)

        iteration += 1

    return best_solution

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

if __name__ == '__main__':
    width = int(input("Urči vertikálnu vzdialenosť: "))
    height = int(input("Urči horizontálnu vzdialenosť: "))
    nodes = generate_random_node_locations(width, height)
    current_node = choose_starting_node(nodes)
    max_iterations = 50
    result = start()
    print()
