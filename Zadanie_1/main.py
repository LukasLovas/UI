from Field import Field


def board_setup():
    size = int(input("Define size: "))
    matrix = [[Field(j, i) for j in range(size)] for i in range(size)]
    for row in matrix:
        for field in row:
            print("(x: " + str(field.x) + " y: " + str(field.y), end=") ")
        print()
    return matrix


def get_starting_position():
    x = int(input("Define starting position (X): "))
    y = int(input("Define starting position (Y): "))
    board[y][x].order_number = 1
    print("Starting position: (x: " + str(x) + " y: " + str(y) + ")")
    return board[y][x]


def calculate_possible_moves(matrix, current_field):
    all_moves = []
    possible_moves = []
    position = (current_field.x, current_field.y)
    for vector in move_vectors:
        all_moves.append(position + vector)
    for move in all_moves:
        if move[0]:
            possible_moves.append(move)
    return possible_moves


if __name__ == "__main__":
    move_vectors = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
    game_cycle = True
    board = board_setup()
    field = get_starting_position()
    order_counter = 1
    while game_cycle:
        calculate_possible_moves(board, field)
        break
