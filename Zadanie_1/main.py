from Field import Field


def board_setup():
    size = int(input("Define size: "))
    matrix = [[Field(j, i) for j in range(size)] for i in range(size)]  # 2 dimensional array serves as a board
    for row in matrix:
        for field in row:
            print("(x: " + str(field.x) + " y: " + str(field.y), end=") ")
        print()
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]  # transpose the matrix


def get_starting_position():
    x = int(input("Define starting position (X): "))
    y = int(input("Define starting position (Y): "))
    board[x][y].order_number = 1  # define as first move
    print("Starting position: (x: " + str(x) + " y: " + str(y) + ")")
    return board[x][y]

def validate_move(move,matrix):
    if 0 <= move[0] < len(matrix) and 0 <= move[1] < len(matrix[0]) and matrix[move[0]][move[1]].check_availability():
        return True
    else:
        return False

def calculate_possible_moves(matrix, current_field):
    all_moves = []
    possible_moves = []
    position = (current_field.x, current_field.y)
    for vector in move_vectors:
        x, y = position[0] + vector[0], position[1] + vector[1]
        all_moves.append((x, y))
    for move in all_moves:
        if validate_move(move,matrix):
            possible_moves.append(move)
    return possible_moves

def move():

if __name__ == "__main__":
    move_vectors = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
    game_cycle = True
    board = board_setup()
    field = get_starting_position()
    order_counter = 1
    while game_cycle:
        move()
        print(calculate_possible_moves(board, field))
        break
