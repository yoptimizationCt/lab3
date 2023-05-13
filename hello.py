class Hello:
    message = "Hello, "

    def __init__(self, name):
        self.message += name

    def say_hello(self):
        print(self.message)
