import app.backend.db.respositories
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.backend.db.respositories import retrieval
from app.backend.model.schemas import Tourism
from sqlalchemy.orm import Session
from app.backend.model import get_db

router = APIRouter()



@router.get('/IR/search', response_class=HTMLResponse)
async def get_retrieve(query, requests: Request, request=Tourism.TourismInRequest, db: Session = Depends(get_db)):
    return retrieval.get_retrieval(
        query=query,
        request=request,
        requests=requests,
        db=db)
    
@router.get('/IR', response_class=HTMLResponse)
async def render(request:Request, db: Session=Depends(get_db)):
    return retrieval.render_all(request, db)