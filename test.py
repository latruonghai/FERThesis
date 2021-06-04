
def test(a, b):
    yield a
    yield b


t = test(1, 2)
for i in list(t):
    print(t)
