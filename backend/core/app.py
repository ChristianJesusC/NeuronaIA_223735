from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import FRONTEND_DIR
from backend.routes import dataset, frontend, prediction, training


def create_app() -> FastAPI:
    app = FastAPI(title="Multilayer Neural Network API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js",  StaticFiles(directory=str(FRONTEND_DIR / "js")),  name="js")

    app.include_router(frontend.router)
    app.include_router(dataset.router)
    app.include_router(training.router)
    app.include_router(prediction.router)

    return app
