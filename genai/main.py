from fastapi import FastAPI
from app.api.routes import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import HTTPException

# Initialize the FastAPI application with a title
app = FastAPI(title="IoT GenAI Gateway")

# Include the API router with a prefix of /v1 for versioning
app.include_router(router, prefix="/v1")

# Add any additional middleware, exception handlers, or startup events here


# Example: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example: Add a startup event
@app.on_event("startup")
async def startup_event():
    print("Application startup complete.")

# Example: Add a custom exception handler

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "custom": "Handled by custom exception handler"},
    )
