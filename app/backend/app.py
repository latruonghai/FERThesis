from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api.routers import api
from .model.base import models
from .model import engine


def generate_app():
    app = FastAPI()

    models.Base.metadata.create_all(engine)
    app.mount(
        '/static',
        StaticFiles(
            directory="app/backend/jinja/static"),
        name="static")
    app.include_router(api.router)
    return app


app = generate_app()
