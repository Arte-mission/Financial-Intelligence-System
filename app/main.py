import logging
from dotenv import load_dotenv

# Ensure environment variables are loaded prior to mounting any backend routes or configuration
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import market, stock
from app.database import engine, Base, SessionLocal
from app.services.ticker_service import seed_companies_from_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created.")
    
    # Seed companies from CSV
    db = SessionLocal()
    try:
        seed_companies_from_csv(db, file_path="company_list.csv")
    finally:
        db.close()
        
    # Start the permanent macro background refresh loop
    logger.info("Initializing background macro loop...")
    market.start_macro_refresh_loop()
        
    yield
    # Shutdown event
    logger.info("Shutting down Nepal Stock Market API.")


app = FastAPI(
    title="Nepal Stock Market API",
    description="A small FastAPI backend for fetching market news sentiment and stock statistics with a validated equity universe.",
    version="1.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend hosting arrays (e.g. VSCode LiveServer on port 5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(market.router)
app.include_router(stock.router)

@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify backend is up and running.
    """
    return {"status": "ok", "message": "Nepal Stock Market API is running smoothly."}
