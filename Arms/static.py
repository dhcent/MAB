#name probably inaccurate, but arm that just returns expected value. Similar to a "state informed" setting.
class StaticArm:
    def __init__(self, mu):
        self.name = "StaticArm"
        self.mean = mu

    def update_val(self, mu):
        self.mean = mu
        return

    def draw(self):
        return self.mean