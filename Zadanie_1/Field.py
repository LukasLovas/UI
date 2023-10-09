class Field:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.order_number = "-"

    def check_availability(self):
        if self.order_number == "-":
            return True
        else:
            return False
