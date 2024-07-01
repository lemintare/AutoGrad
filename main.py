class Var:
    def __init__(self, value, grad=0):
        self.value = value
        self.grad = grad

class Operation:
    def __init__(self):
        self.result = None

    def forward(self):
        pass

    def backward(self, last_grad):
        pass

class Mul(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.result = self.forward()

    def forward(self):
        x_value = self.x.value if isinstance(self.x, Var) else self.x.result if isinstance(self.x, Operation) else self.x
        y_value = self.y.value if isinstance(self.y, Var) else self.y.result if isinstance(self.y, Operation) else self.y
        self.result = x_value * y_value
        return self.result

    def backward(self, last_grad):
        x_grad = self.y.value if isinstance(self.y, Var) else self.y.result if isinstance(self.y, Operation) else self.y
        y_grad = self.x.value if isinstance(self.x, Var) else self.x.result if isinstance(self.x, Operation) else self.x

        if isinstance(self.x, Var):
            self.x.grad += x_grad * last_grad
        if isinstance(self.x, Operation):
            self.x.backward(x_grad * last_grad)

        if isinstance(self.y, Var):
            self.y.grad += y_grad * last_grad
        if isinstance(self.y, Operation):
            self.y.backward(y_grad * last_grad)

def autograd(y):
    y.backward(1)


x1 = Var(2)
x2 = Var(3)

z1 = Mul(x1, 3)
z2 = Mul(x2, 5)
z3 = Mul(z1, z2)
y = Mul(z3, 3)


autograd(y)

print(x1.grad, x2.grad)
