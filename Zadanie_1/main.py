from Field import Field
import time
import random


def board_setup():
    size = int(input("Definuj veľkosť šachovnice: "))
    matrix = [[Field(j, i) for j in range(size)] for i in range(size)]  # Šachovnica (dvojdimenzionálna matica)
    for row in matrix:
        for field in row:
            print("(x: " + str(field.x) + " y: " + str(field.y), end=") ")  # výpis matice
        print()
    return matrix


def get_starting_position(board):
    x = int(input("Definuj štartovaciu pozíciu (X): "))  # Definovanie štartovacej pozície na šachovnici
    y = int(input("Definuj štartovaciu pozíciu (Y): "))
    board[x][y].order_number = 1  # define as first move
    print("Štartovacia pozícia: (x: " + str(x) + " y: " + str(y) + ")")
    return board[x][y]


def validate_move(move, matrix):
    if 0 <= move[0] < len(matrix) and 0 <= move[1] < len(matrix[0]) and matrix[move[1]][
        move[0]].check_availability():  # Podmienka na validovanie kroku, overuje,
        # či vypočítaný krok nie je mimo šachovnice alebo či už na políčku kôň nebol
        return True
    else:
        return False


def calculate_possible_moves(matrix, current_field):
    possible_moves = []
    position = (current_field.x, current_field.y)
    for vector in move_vectors:
        (x, y) = (position[0] + vector[0], position[1] + vector[1])  # Počítanie všetkých možných krokov
        if validate_move((x, y), matrix):
            possible_moves.append(vector)
    return possible_moves


def generate_5_random_positions(board):
    positions = [(0, len(board[0]) - 1)]
    counter = 1
    while counter != 5:
        x = random.randint(0, len(board) - 1)  # Generovanie 5 náhodných pozícií poďľa zadaného argumentu (board5x5
        # alebo board6x6)
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
                                           # Funkcia pomocných módov programu
                                           start_time)
    if result is not False:
        print("Riešenie existuje")
    else:
        print("Riešenie neexistuje")


def reset_board(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            board[i][j].order_number = "-"  # Vyčistenie šachovnice
    return board


def start_task_f():
    order_counter = 1
    moves = 0
    start_time = time.time()
    board_5x5 = [[Field(j, i) for j in range(5)] for i in range(5)]  # Vytvorenie šachovnice
    positions = generate_5_random_positions(board_5x5)  # Vytvorenie náhodných súradníc
    for position in positions:
        current_field = board_5x5[position[1]][position[0]]  # Vyhľadanie momentálneho políčka (objekt)
        current_field.order_number = 1  # Nastaviť poradové číslo začiatočného políčka
        result = move_simple(board_5x5, current_field, calculate_possible_moves(board_5x5, current_field),
                             # rekurzívna funkcia (DFS)
                             order_counter,
                             moves,
                             start_time)
        if result is not False:
            print("Riešenie existuje")
            dummy_matrix = [[board_5x5[j][i].order_number for j in range(len(board_5x5[0]))] for i in
                            # Šachovnica s poradovými číslami
                            range(len(board_5x5[0]))]
            order_number_matrix = [[dummy_matrix[j][i] for j in range(len(dummy_matrix))] for i in
                                   range(len(dummy_matrix[0]))]  # transponovanie
            for row in order_number_matrix:
                for item in row:
                    print(f"{item:>{3}}", end=" ")  # Výpis šachovnice
                print()
            time_till_completion = time.time() - start_time
            print("Štartovacia pozícia: (" + str(current_field.x) + "," + str(current_field.y) + ")")  # Výpis detailov
            print("Riešenie nájdené: " + str(True))
            print("Čas: " + str(time_till_completion))
            print("Počet krokov: " + str(result))
            print("------------------------------\n")
        else:
            print("Pre pozíciu (" + str(current_field.x),
                  str(current_field.y) + ") na hracej ploche 5x5 neexistuje riešenie.")  # Oznámenie neúspešnosti programu
            print("-------------------------")
        board_5x5 = reset_board(board_5x5)  # Resetovanie šachovnice pre ďaľšie štartovacie súradnice
    print("\n\nŠachovnice 6x6\n\n")
    order_counter = 1
    moves = 0  # Pre šachovnice 6x6 je dokumentácia rovnaká ako pre 5x5
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
            print("Riešenie existuje\n")
            time_till_completion = time.time() - start_time
            dummy_matrix = [[board_6x6[j][i].order_number for j in range(len(board_6x6[0]))] for i in
                            range(len(board_6x6[0]))]
            order_number_matrix = [[dummy_matrix[i][j] for j in range(len(dummy_matrix))] for i in
                                   range(len(dummy_matrix[0]))]
            for row in order_number_matrix:
                for item in row:
                    print(f"{item:>{3}}", end=" ")
                print()
            print("Štartovacia pozícia: (" + str(current_field.x) + "," + str(current_field.y) + ")")
            print("Riešenie nájdené: " + str(True))
            print("Čas: " + str(time_till_completion))
            print("Počet krokov: " + str(result))
            print("------------------------------\n")
            board_6x6 = reset_board(board_6x6)
        else:
            print("Pre pozíciu (" + str(current_field.x),
                  str(current_field.y) + ") na hracej ploche 6x6 neexistuje riešenie.")


def move_timer_stepper_move_limit(board, current_field, move_vectors, order_counter, moves,
                                  start_time):  # Funkcia pre pomocné módy programu
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
                    if input("Pokračuj?: (enter)") != "":
                        exit()
                    result = move_timer_stepper_move_limit(board, new_field, calculate_possible_moves(board, new_field),
                                                           order_counter + 1, moves,
                                                           start_time)
                    if result is not False:
                        return moves + result
                    new_field.order_number = "-"  # Backtrack
                    moves += 1
                    print("Iterácia zlyhala pre " + str(
                        (new_field.x, new_field.y)) + ", vraciam sa na číslo: " + str(
                        order_counter) + ", a číslo " + str(order_counter + 1) + " dávam na ďaľšiu možnú pozíciu.")

        return False
    else:
        print("Maximálny počet krokov prekročený, program končí.")
        return False


def move_simple(board, current_field, move_vectors, order_counter, moves, start_time):
    time_elapsed = time.time() - start_time  # Meranie času
    global flag  # Príznak pre časový limit
    if time_elapsed >= TIME_LIMIT and flag == 0:  # Prekročenie časového limitu
        print("Čas vypršal pre pozíciu (" + str(current_field.x) + "," + str(current_field.y) + "), program končí")
        print("------------------------------\n")
        flag = 1
        return False

    moves += 1  # Pridanie jednoho kroku
    if moves <= MAX_MOVES:  # Overenie pre limit krokov
        if order_counter == len(board) * len(
                board[0]):  # Overenie, či program neskončil vyhľadávanie a nemá už celé riešenie
            return moves

        if not move_vectors:  # Overenie pre prázdne pole možných krokov
            return False

        for vector in move_vectors:  # Pre každý vektor rekurzívne prehľadaj celú vetvu stromu
            new_coords = (current_field.x + vector[0], current_field.y + vector[1])  # Zadanie nových súradníc po pohybe
            if validate_move(new_coords, board):  # Validovanie kroku
                new_field = board[new_coords[1]][new_coords[0]]  # Premenná pre pole šachovnice s novými súradnicami
                if new_field.order_number == "-":  # Dodatočné overenie
                    new_field.order_number = order_counter + 1  # Pridanie poradového čísla +1
                    result = move_simple(board, new_field, calculate_possible_moves(board, new_field),  # Rekurzívna funkcia
                                         order_counter + 1, moves,
                                         start_time)
                    if result is not False:  # Pokiaľ sa z ďalšej iterácie algoritmu nevrátila boolean hodnota False, znamená to že sa našlo riešenie
                        return moves + result  # Zober počet krokov ktorý vykonali kroky predomnou a pripočítaj ich k počtu mojich krokov, a spätnou rekurziou ich vráť
                    new_field.order_number = "-"  # Backtrackovanie políčka šachovnice pri neúspešnom vyhľadaní
                    moves += 1  # Backtracking počítam tiež ako krok, aj keď krok opravný
        return False  # Pri nenájdení vráť False
    else:
        print("Maximálny počet krokov dosiahnutý. Program končí.")
        return False


def start_normal_evaluation():  # Funkcia pre pomocné módy
    start_time = time.time()
    board = board_setup()
    field = get_starting_position(board)
    result = move_just_get_result_single_problem(board, field, calculate_possible_moves(board, field), order_counter,
                                                 moves,
                                                 start_time)
    if result is not False:
        print("Riešenie existuje")
        dummy_matrix = [[board[j][i].order_number for j in range(len(board[0]))] for i in
                        range(len(board[0]))]
        order_number_matrix = [[dummy_matrix[i][j] for j in range(len(dummy_matrix))] for i in
                               range(len(dummy_matrix[0]))]  # transpose this matrix as well
        for row in order_number_matrix:
            for item in row:
                print(f"{item:>{3}}", end=" ")
            print()

    else:
        print("Riešenie neexistuje")


def move_just_get_result_single_problem(board, current_field, move_vectors, order_counter, moves,
                                        start_time):  # Funkcia pre pomocné módy
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
                    result = move_just_get_result_single_problem(board, new_field,
                                                                 calculate_possible_moves(board, new_field),
                                                                 order_counter + 1, moves,
                                                                 start_time)
                    if result is not False:
                        return moves + result
                    new_field.order_number = "-"  # Backtrack
                    moves += 1

        return False
    else:
        print("Maximálny počet krokov dosiahnutý. Program končí.")
        return False


if __name__ == "__main__":
    TIME_LIMIT = 15  # Časový limit v sekundách
    MAX_MOVES = 50000  # Počet krokov limit
    move_vectors = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1),
                    (-1, 2)]  # Reprezentácia pohybov vektormi
    order_counter = 1  # Poradové číslo
    flag = 0
    moves = 0  # Počet krokov
    print("Pre riešenie problému s pomocným krokovaním a výpiskami krokov, stlačte \"1\"\n"
          "Pre riešenie Zadania f)(5 náhodných pozícií na šachovnici 5x5,5 na 6x6), stlačte \"2\"\n"  # Základný print do konzole na výber módu a následne input pre výber.
          "Pre rýchle vyriešenie problému len s výsledkom, stlačte \"3\"")
    program_type = input("Vyber mód programu: ")
    if program_type == "1":
        start_with_move_and_printing_and_stepper()
    elif program_type == "2":
        start_task_f()
    elif program_type == "3":
        start_normal_evaluation()
