class Field:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.order_number = None

    def check_availability(self):
        if self.order_number is None:
            return True
        else:
            return False
