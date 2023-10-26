from random import randint
from Node import Node
from math import hypot


def generate_random_location(width, height):
    x = randint(0, width)
    y = randint(0, height)
    return x, y


def generate_random_node_locations(width, height):
    n = int(input("Urči počet miest: "))
    used = []
    nodes = []
    for i in range(n):
        x, y = generate_random_location(width, height)
        if (x, y) not in used:
            used.append((x, y))
            nodes.append(Node(x, y, 0))
    return nodes


def choose_starting_node(nodes):
    node = None
    for i in range(len(nodes)):
        print(f"No. {i+1}: (x: {nodes[i].x} y: {nodes[i].y})")
    while node is None:
        start = int(input("Vyber poradové číslo mesta, z ktorého začať: "))
        if 0 < start < len(nodes):
            node = nodes[start]
            node.order = 1
    return node


if __name__ == '__main__':
    width = int(input("Urči vertikálnu vzdialenosť: "))
    height = int(input("Urči horizontálnu vzdialenosť: "))
    nodes = generate_random_node_locations(width, height)
    starting_node = choose_starting_node(nodes)
    print()
