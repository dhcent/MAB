#name probably inaccurate, but arm that just returns expected value. Similar to a "state informed" setting.
class NormalArm:
    def __init__(self, mu):
        self.mean = mu

    def update_val(self, mu):
        self.mean = mu
        return

    def draw(self):
        return self.mean