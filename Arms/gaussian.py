class GaussianArm:
    def __init__(self, expected, std_dev=0.1):
        self.expected_val = expected
        self.std_dev = std_dev

    def update_val(self, new_expected):
        self.expected_val = new_expected
        return

    def draw(self):
        return np.random.normal(self.expected_val, self.std_dev)