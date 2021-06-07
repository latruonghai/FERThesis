from fastapi import APIRouter
from fastapi import Request
from app.backend.db.respositories import hellman

router = APIRouter()

@router.get('/hellman')
def render(request: Request):
    return hellman.render("main21.html", request)

@router.get('/hellman/compute')
def process(
    a: int,
    b: int,
    module: int,
    basex: int,
    basey: int,
    aliceSecretKey: int,
        bobSecretKey: int):

    return hellman.process(
        a,
        b,
        module,
        basex,
        basey,
        aliceSecretKey,
        bobSecretKey)
