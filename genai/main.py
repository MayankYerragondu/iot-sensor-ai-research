from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="IoT GenAI Gateway")

app.include_router(router, prefix="/v1")