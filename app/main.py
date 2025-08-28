from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, store, suitability, facilities, regions, buffer

app = FastAPI(title="Suitability API", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(suitability.router, prefix="/suitability", tags=["Suitability"])
app.include_router(facilities.router, prefix="/facilities", tags=["Facilities"])
app.include_router(regions.router, prefix="/regions", tags=["Regions"])
app.include_router(buffer.router, prefix="/buffer", tags=["Buffer"])
app.include_router(store.router, prefix="/store", tags=["Store"])