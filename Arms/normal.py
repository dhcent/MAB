class NormalArm:
    def __init__(self, expected):
        self.expected_val = expected

    def update_val(self, new_expected):
        self.expected_val = new_expected
        return

    def draw(self):
        return self.expected_val