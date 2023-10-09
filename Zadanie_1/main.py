from Field import Field


def board_setup():
    size = int(input("Define size: "))
    matrix = [[Field(j, i) for j in range(size)] for i in range(size)]  # 2 dimensional array serves as a board
    for row in matrix:
        for field in row:
            print("(x: " + str(field.x) + " y: " + str(field.y), end=") ")
        print()
    return matrix

def get_starting_position():
    x = int(input("Define starting position (X): "))
    y = int(input("Define starting position (Y): "))
    board[x][y].order_number = 1  # define as first move
    print("Starting position: (x: " + str(x) + " y: " + str(y) + ")")
    return board[x][y]


def validate_move(move, matrix):
    if 0 <= move[0] < len(matrix) and 0 <= move[1] < len(matrix[0]) and matrix[move[0]][move[1]].check_availability():
        return True
    else:
        return False


def calculate_possible_moves(matrix, current_field):
    possible_moves = []
    position = (current_field.x, current_field.y)
    for vector in move_vectors:
        (x, y) = (position[0] + vector[0], position[1] + vector[1])
        if validate_move((x, y), matrix):
            possible_moves.append(vector)
    return possible_moves


def move(board, current_field, move_vectors, order_counter):
    if order_counter == len(board) * len(board[0]):
        return True

    for vector in move_vectors:
        new_coords = (current_field.x + vector[0], current_field.y + vector[1])
        if validate_move(new_coords, board):
            new_field = board[new_coords[0]][new_coords[1]]
            if new_field.order_number == "-":
                new_field.order_number = order_counter + 1
                dummy_matrix = [[board[j][i].order_number for j in range(len(board[0]))] for i in range(len(board[0]))]
                order_number_matrix = [[dummy_matrix[i][j] for j in range(len(dummy_matrix))] for i in range(len(dummy_matrix[0]))]  # transponse this matrix as well
                for row in order_number_matrix:
                    for item in row:
                        print(f"{item:>{3}}", end=" ")
                    print()
                print("\n----------------------------------\n")
                if move(board, new_field, move_vectors, order_counter + 1):
                    return True
                new_field.order_number = "-"  # Backtrack

    return False


if __name__ == "__main__":
    move_vectors = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
    game_cycle = True
    board = board_setup()
    field = get_starting_position()
    order_counter = 1

    if move(board, field, calculate_possible_moves(board, field), order_counter):
        print("completed")
    else:
        print("fucked up")
