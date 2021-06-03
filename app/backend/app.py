from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api.routers import api


def generate_app():
    app = FastAPI()
    app.mount('/static', StaticFiles(directory="app/backend/jinja/static"), name="static")
    app.include_router(api.router)
    return app


app = generate_app()
