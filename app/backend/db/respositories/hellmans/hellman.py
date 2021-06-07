from app.backend.db.respositories.hellmans.elliptic import *
from app.backend.db.respositories.hellmans.finitefield.finitefield import FiniteField


import os


class Hellman():
    def __init__(self, a, b, x, y, module):
        self.F = FiniteField(module, 1)
        self.curve = EllipticCurve(a, b)
        self.basepoint = Point(self.curve, self.F(x), self.F(y))
    def generateSecretKey(self, numBits):
        return int.from_bytes(os.urandom(numBits // 8), byteorder='big')

    def sendDH(self, privateKey, generator, sendFunction):
        return sendFunction(privateKey * generator)

    def receiveDH(self, privateKey, receiveFunction):
        return privateKey * receiveFunction()

    def slowOrder(self, point):
        Q = point
        i = 1

        while True:
            if isinstance(Q, Ideal):
                return i
            else:
                Q = Q + point
                i += 1


if __name__ == "__main__":

    # Totally insecure curve: y^2 = x^3 + 324x + 1287
    print(" y^2 = x^3 + ax + b")
    print("a,b : ...")
    a, b = map(int, input().split())

    print("Module: ")
    p = int(input())
    F = FiniteField(p, 1)
    curve = EllipticCurve(a=F(a), b=F(b))
    # order is 1964
    print("basepoint: ..")
    coor_x, coor_y = map(int, input().split())
    basePoint = Point(curve, F(coor_x), F(coor_y))

    print("alice private key: ..")
    aliceSecretKey = int(input())

    print("bob private key: ..")
    bobSecretKey = int(input())

    # Output

    print('Secret keys are %d, %d' % (aliceSecretKey, bobSecretKey))
    alicePublicKey = sendDH(aliceSecretKey, basePoint, lambda x: x)
    bobPublicKey = sendDH(bobSecretKey, basePoint, lambda x: x)

    print('alice public key {} - bob public key {}  '.format(alicePublicKey, bobPublicKey))
    sharedSecret1 = receiveDH(bobSecretKey, lambda: alicePublicKey)
    sharedSecret2 = receiveDH(aliceSecretKey, lambda: bobPublicKey)

    print('Shared secret is %s == %s' % (sharedSecret1, sharedSecret2))
    print(
        'extracing x-coordinate to get an integer shared secret: %d' %
        (sharedSecret1.x.n))
    
