class progress():
    def __init__(self, length):
        self.length = length
        self.bar = ['|'] + [' '] * length + ['|']
        self.head = 0
        self.end = False
        self.ending = ''
        
    def start(self):
        self.bar = ['|'] + [' '] * self.length + ['|']
        self.end = False

    def update(self, ex_info:str = ''):
        self.ending = ex_info
        self.bar[self.head + 1] = '█'
        if self.head != self.length:
            self.head += 1

    def __repr__(self):
        if self.head != self.length:
            if self.ending != '':
                return ''.join(self.bar) + " - " + self.ending + "\033[A"
            return ''.join(self.bar) + "\033[A"
        
        if self.end == False:
            self.end = True
            return f"{''.join(self.bar)} - Done!"
        return "\033[A"
