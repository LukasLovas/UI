from Field import Field


def board_setup():
    rows = int(input("Define number of rows: ")) or 0
    columns = int(input("Define number of columns: ")) or 0
    matrix = []
    for i in range(rows):
        column = []
        for j in range(columns):
            field = Field(j, i)
            column.append(field)
        matrix.append(column)
    for row in matrix:
        print()
        for field in row:
            print("(x: " + str(field.x) + " y: " + str(field.y), end=") ")
    print()
    return matrix


def get_starting_position():
    x = int(input("Input row starting position (X): "))
    y = int(input("Input column starting position (Y): "))
    board[x][y].order_number = 1
    print("Starting position: (" + str(x) + " " + str(y) + ")")
    return board[x][y]


def calculate_possible_moves(matrix, current_field):
    # rru,rrd,ddr,ddl,llu,lld,uur,uul
    all_moves = []
    possible_moves = []
    rru = matrix[current_field.x + 2][current_field.y - 1] or None
    all_moves.append(rru)
    rrd = matrix[current_field.x + 2][current_field.y + 1] or None
    all_moves.append(rrd)
    ddr = matrix[current_field.x + 1][current_field.y + 2] or None
    all_moves.append(ddr)
    ddl = matrix[current_field.x - 1][current_field.y + 2] or None
    all_moves.append(ddl)
    llu = matrix[current_field.x - 2][current_field.y - 1] or None
    all_moves.append(llu)
    lld = matrix[current_field.x - 2][current_field.y + 1] or None
    all_moves.append(lld)
    uur = matrix[current_field.x + 1][current_field.y - 2] or None
    all_moves.append(uur)
    uul = matrix[current_field.x - 1][current_field.y + 2] or None
    all_moves.append(uul)
    for move in all_moves:
        if move.check_availability() and move is not None:
            possible_moves.append(move)
    return possible_moves


if __name__ == "__main__":
    game_cycle = True
    board = board_setup()
    field = get_starting_position()
    order_counter = 1
    while game_cycle:
        calculate_possible_moves(board, field)
