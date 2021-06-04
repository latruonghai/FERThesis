from app.backend.db.query.okapi import Okapi
import pandas as pd
from app.backend.model.base.models import Tourism
from fastapi.responses import RedirectResponse
from app.backend.model.schemas.Tourism import TourismInRequest
from app.backend.model import get_db
from app.backend.core import TEMPLATE
from app.backend.model.OkapiBM25.loadmodel import FILEPATH
from time import time


que = Okapi(FILEPATH)

def get_retrieval(
        query: str,
        request,
        requests,
        db):
    
    data = db.query(Tourism)
    start = time()
    # que = Okapi(query, file_path)
    KW = que.preprocessing_query(query)
    kw = str(KW)
    if CheckValid(data, kw):
        return render(requests, db, kw, start)
    else:
        df = pd.read_csv('Source_new.csv')
        res, name = que.letQuery(KW)
        try:
            for r, n in zip(res, name):
                sources = df[df['Files name'] == n]['Sources'].tolist()[0]

                new_post = request(
                    title=str(r['title']),
                    content=str(r['content']),
                    keyword=kw,
                    source=str(sources)
                )

                create_data(request=new_post, db=db)

        except(TypeError, IndexError):
            data.delete(synchronize_session=False)
            db.commit()
            return RedirectResponse(f'/IR')

        all_posts = data.filter(
            kw == Tourism.keyword).order_by(
            Tourism.ids).all()
        end = time() - start
        return TEMPLATE.TemplateResponse(
            "query12.html", {"request": requests, "posts": all_posts, "time": round(end, 4)})


def create_data(request, db):
    new_tour = Tourism(
        title=request.title,
        content=request.content,
        keyword=request.keyword,
        source=request.source
    )
    db.add(new_tour)
    db.commit()
    db.refresh(new_tour)


def render_all(request, db):
    all_data = db.query(Tourism).all()
    return TEMPLATE.TemplateResponse(
        "query.html", {
            "request": request, "posts": all_data})


def CheckValid(data, val):
    dt = data.filter(val == Tourism.keyword).all()
    if not dt:
        return False
    else:
        return True


def render(request, db, kw, times):
    all_data = db.query(Tourism).filter(Tourism.keyword == kw).all()
    end = time() - times
    return TEMPLATE.TemplateResponse(
        "query12.html", {
            "request": request, "posts": all_data, "time": round(end, 4)})
