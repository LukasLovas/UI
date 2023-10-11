from Field import Field
import time
import random

def board_setup():
    size = int(input("Define size: "))
    matrix = [[Field(j, i) for j in range(size)] for i in range(size)]  # 2 dimensional array serves as a board
    for row in matrix:
        for field in row:
            print("(x: " + str(field.x) + " y: " + str(field.y), end=") ")
        print()
    return matrix


def get_starting_position(board):
    x = int(input("Define starting position (X): "))
    y = int(input("Define starting position (Y): "))
    board[x][y].order_number = 1  # define as first move
    print("Starting position: (x: " + str(x) + " y: " + str(y) + ")")
    return board[x][y]


def validate_move(move, matrix):
    if 0 <= move[0] < len(matrix) and 0 <= move[1] < len(matrix[0]) and matrix[move[1]][move[0]].check_availability():
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


def generate_5_random_positions(board):
    positions = [(0, len(board[0]) - 1)]
    counter = 1
    while counter != 5:
        x = random.randint(0, len(board) - 1)
        y = random.randint(0, len(board) - 1)
        if (x, y) not in positions:
            positions.append((x, y))
            counter += 1
    return positions


def start_with_move_and_printing_and_stepper():
    start_time = time.time()
    board = board_setup()
    field = get_starting_position(board)
    result = move_timer_stepper_move_limit(board, field, calculate_possible_moves(board, field), order_counter, moves,
                                           start_time)
    if result is not False:
        print("Riesenie existuje")
    else:
        print("Riesenie neexistuje")


def reset_board(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            board[i][j].order_number = "-"
    return board


def start_task_f():
    order_counter = 1
    moves = 0
    start_time = time.time()
    board_5x5 = [[Field(j, i) for j in range(5)] for i in range(5)]
    positions = generate_5_random_positions(board_5x5)
    for position in positions:
        current_field = board_5x5[position[1]][position[0]]
        current_field.order_number = 1
        result = move_simple(board_5x5, current_field, calculate_possible_moves(board_5x5, current_field),
                             order_counter,
                             moves,
                             start_time)
        if result is not False:
            print("Riesenie existuje")
            dummy_matrix = [[board_5x5[j][i].order_number for j in range(len(board_5x5[0]))] for i in
                            range(len(board_5x5[0]))]
            order_number_matrix = [[dummy_matrix[j][i] for j in range(len(dummy_matrix))] for i in
                                   range(len(dummy_matrix[0]))]  # transpose this matrix as well
            for row in order_number_matrix:
                for item in row:
                    print(f"{item:>{3}}", end=" ")
                print()
            time_till_completion = time.time() - start_time
            print("Starting position: (" + str(current_field.x) + "," + str(current_field.y) + ")")
            print("Solution found: " + str(True))
            print("Time: " + str(time_till_completion))
            print("Moves: " + str(result))
            print("------------------------------\n")
        else:
            print("Pre poziciu (" + str(current_field.x),
                  str(current_field.y) + ") na hracej ploche 5x5 neexistuje riesenie.")
            print("-------------------------")
        board_5x5 = reset_board(board_5x5)
    print("\n\nMoving on to 6x6\n\n")
    order_counter = 1
    moves = 0
    start_time = time.time()
    board_6x6 = [[Field(j, i) for j in range(6)] for i in range(6)]
    positions = generate_5_random_positions(board_6x6)
    for position in positions:
        current_field = board_6x6[position[1]][position[0]]
        current_field.order_number = 1
        result = move_simple(board_6x6, current_field, calculate_possible_moves(board_6x6, current_field),
                             order_counter,
                             moves,
                             start_time)
        if result is not False:
            print("Riesenie existuje\n")
            time_till_completion = time.time() - start_time
            dummy_matrix = [[board_6x6[j][i].order_number for j in range(len(board_6x6[0]))] for i in
                            range(len(board_6x6[0]))]
            order_number_matrix = [[dummy_matrix[i][j] for j in range(len(dummy_matrix))] for i in
                                   range(len(dummy_matrix[0]))]
            for row in order_number_matrix:
                for item in row:
                    print(f"{item:>{3}}", end=" ")
                print()
            print("Starting position: (" + str(current_field.x) + "," + str(current_field.y) + ")")
            print("Solution found: " + str(True))
            print("Time: " + str(time_till_completion))
            print("Moves: " + str(result))
            print("------------------------------\n")
            board_6x6 = reset_board(board_6x6)
        else:
            print("Pre poziciu (" + str(current_field.x),
                  str(current_field.y) + ") na hracej ploche 6x6 neexistuje riesenie.")


def move_timer_stepper_move_limit(board, current_field, move_vectors, order_counter, moves, start_time):
    time_elapsed = time.time() - start_time
    moves += 1
    if moves <= MAX_MOVES:
        if order_counter == len(board) * len(board[0]):
            return moves

        if not move_vectors:
            return False

        for vector in move_vectors:
            new_coords = (current_field.x + vector[0], current_field.y + vector[1])
            if validate_move(new_coords, board):
                new_field = board[new_coords[1]][new_coords[0]]
                if new_field.order_number == "-":
                    new_field.order_number = order_counter + 1
                    dummy_matrix = [[board[j][i].order_number for j in range(len(board[0]))] for i in
                                    range(len(board[0]))]
                    order_number_matrix = [[dummy_matrix[i][j] for j in range(len(dummy_matrix))] for i in
                                           range(len(dummy_matrix[0]))]  # transpose this matrix as well
                    for row in order_number_matrix:
                        for item in row:
                            print(f"{item:>{3}}", end=" ")
                        print()
                    print("\n----------------------------------\n")
                    if input("continue?: ") != "":
                        exit()
                    result = move_timer_stepper_move_limit(board, new_field, calculate_possible_moves(board, new_field),
                                                           order_counter + 1, moves,
                                                           start_time)
                    if result is not False:
                        return moves + result
                    new_field.order_number = "-"  # Backtrack
                    moves += 1
                    print("Iteration failed for " + str(
                        (new_field.x, new_field.y)) + ", backtracking to order number: " + str(
                        order_counter) + ", number " + str(order_counter + 1) + " placed elsewhere.")

        return False
    else:
        print("Maximum move attempts exceeded, stopping the program.")
        return False


def move_simple(board, current_field, move_vectors, order_counter, moves, start_time):
    time_elapsed = time.time() - start_time
    global flag
    if time_elapsed >= TIME_LIMIT and flag == 0:
        print("Timer ran out, stopping the program.")
        print("------------------------------\n")
        flag = 1
        return False

    moves += 1
    if moves <= MAX_MOVES:
        if order_counter == len(board) * len(board[0]):
            return moves

        if not move_vectors:
            return False

        for vector in move_vectors:
            new_coords = (current_field.x + vector[0], current_field.y + vector[1])
            if validate_move(new_coords, board):
                new_field = board[new_coords[1]][new_coords[0]]
                if new_field.order_number == "-":
                    new_field.order_number = order_counter + 1
                    result = move_simple(board, new_field, calculate_possible_moves(board, new_field),
                                         order_counter + 1, moves,
                                         start_time)
                    if result is not False:
                        return moves + result
                    new_field.order_number = "-"  # Backtrack
                    moves += 1
        return False
    else:
        print("Maximum move attempts exceeded, stopping the program.")
        return False

def start_normal_evaluation():
    start_time = time.time()
    board = board_setup()
    field = get_starting_position(board)
    result = move_just_get_result_single_problem(board, field, calculate_possible_moves(board, field), order_counter, moves,
                                           start_time)
    if result is not False:
        print("Riesenie existuje")
        dummy_matrix = [[board[j][i].order_number for j in range(len(board[0]))] for i in
                        range(len(board[0]))]
        order_number_matrix = [[dummy_matrix[i][j] for j in range(len(dummy_matrix))] for i in
                               range(len(dummy_matrix[0]))]  # transpose this matrix as well
        for row in order_number_matrix:
            for item in row:
                print(f"{item:>{3}}", end=" ")
            print()

    else:
        print("Riesenie neexistuje")


def move_just_get_result_single_problem(board, current_field, move_vectors, order_counter, moves, start_time):
    time_elapsed = time.time() - start_time
    moves += 1
    if moves <= MAX_MOVES:
        if order_counter == len(board) * len(board[0]):
            return moves

        if not move_vectors:
            return False

        for vector in move_vectors:
            new_coords = (current_field.x + vector[0], current_field.y + vector[1])
            if validate_move(new_coords, board):
                new_field = board[new_coords[1]][new_coords[0]]
                if new_field.order_number == "-":
                    new_field.order_number = order_counter + 1
                    result = move_just_get_result_single_problem(board, new_field, calculate_possible_moves(board, new_field),
                                                           order_counter + 1, moves,
                                                           start_time)
                    if result is not False:
                        return moves + result
                    new_field.order_number = "-"  # Backtrack
                    moves += 1


        return False
    else:
        print("Maximum move attempts exceeded, stopping the program.")
        return False


if __name__ == "__main__":
    TIME_LIMIT = 15
    MAX_MOVES = 50000
    move_vectors = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    order_counter = 1
    flag = 0
    moves = 0
    print("For solving a problem with a stepper and printouts, choose \"1\"\n"
          "For solving task F (5 random positions in 5x5 and 6x6), choose \"2\"\n"
          "For solving a single problem with just the result, choose \"3\"")
    program_type = input("Choose: ")
    if program_type == "1":
        start_with_move_and_printing_and_stepper()
    elif program_type == "2":
        start_task_f()
    elif program_type == "3":
        start_normal_evaluation()
