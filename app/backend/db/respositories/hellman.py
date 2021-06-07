from app.backend.db.respositories.hellmans.hellman import Hellman
from app.backend.core import TEMPLATE
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def render(html: str, request):
    return TEMPLATE.TemplateResponse(html, {'request': request})

def process(
    a: int,
    b: int,
    module: int,
    basex: int,
    basey: int,
    aliceSecretKey: int,
        bobSecretKey: int):

    hellman = Hellman(a, b, basex, basey, module)
    d = {}
    d["Secret keys are"], d["Publickey"], d["Share secret Key"], d["Result"], d["Print Result"] = {}, {}, {}, {}, {}
    d["Secret keys are"]['Bob'] = bobSecretKey
    d["Secret keys are"]["Alice"] = aliceSecretKey
    
    alicePublicKey = hellman.sendDH(aliceSecretKey, hellman.basepoint, lambda x: x)
    bobPublicKey = hellman.sendDH(bobSecretKey, hellman.basepoint, lambda x: x)
    
    
    d["Publickey"]["Bob"] = str(bobPublicKey)
    d["Publickey"]["Alice"] = str(alicePublicKey)
    
    shareSecretKey1 = hellman.receiveDH(bobSecretKey, lambda: alicePublicKey)
    shareSecretKey2 = hellman.receiveDH(aliceSecretKey, lambda: bobPublicKey)
    
    d['Share secret Key']["Bob"] = str(shareSecretKey1)
    d['Share secret Key']["Alice"] = str(shareSecretKey2)
    
    d["Result"]["Bob"] = f"extracing x-coordinate to get an integer shared secret: {shareSecretKey1.x.n}"
    d["Result"]["Alice"] = f"extracing x-coordinate to get an integer shared secret: {shareSecretKey2.x.n}"
    
    d["Print Result"]["Bob"] = f'({shareSecretKey1.x.n}, {shareSecretKey1.y.n})'
    d["Print Result"]["Alice"] = f'({shareSecretKey2.x.n}, {shareSecretKey2.y.n})'
    print(d)
    return JSONResponse(content=jsonable_encoder(d))
