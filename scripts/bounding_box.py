class BoundingBox:
    def __init__(self, x=0, y=0, w=1, h=1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def update(self, x=None, y=None, w=None, h=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if w is not None:
            self.w = w
        if h is not None:
            self.h = h

    def to_list(self):
        return [self.x, self.y, self.w, self.h]
