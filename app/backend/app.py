from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api.routers import api
from .model.base import models
from .model import engine
from fastapi.middleware.cors import CORSMiddleware


def generate_app():
    app = FastAPI()

    origins = [
    "http://localhost:3000",
    "localhost:3000"
    ]


    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    models.Base.metadata.create_all(engine)
    app.mount(
        '/static',
        StaticFiles(
            directory="app/backend/jinja/static"),
        name="static")
    app.include_router(api.router)
    return app


app = generate_app()
