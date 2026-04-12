import csv
import os
import re
import logging
from sqlalchemy.orm import Session
from app.models import Company

logger = logging.getLogger(__name__)

def is_equity(symbol: str, sector: str, name: str) -> bool:
    """
    Filter only equity stocks. 
    Include rows unless they clearly represent debentures, bonds, mutual funds, or debt instruments.
    """
    symbol_lower = symbol.lower()
    sector_lower = sector.lower()
    name_lower = name.lower()
    
    if "/" in symbol:
        return False
        
    # Check for year-like patterns 2070 to 2099 etc.
    if re.search(r'20\d{2}', symbol):
        return False
    
    invalid_terms = ["debenture", "bond", "mutual fund"]
    for term in invalid_terms:
        if term in sector_lower or term in name_lower:
            return False
            
    return True

def seed_companies_from_csv(db: Session, file_path: str = "company_list.csv"):
    """
    Reads company_list.csv and seeds the `companies` table.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found. Skipping company seeding.")
        return
        
    # Check if we already have records
    existing_count = db.query(Company).count()
    if existing_count > 0:
        logger.info(f"Companies table already has {existing_count} records. Reseeding skipped.")
        return

    try:
        # utf-8-sig automatically handles the byte order mark (BOM) if exported from Excel
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            total_rows = 0
            kept_rows = 0
            excluded_rows = 0
            first_10 = []
            
            for row in reader:
                total_rows += 1
                
                symbol = str(row.get("Symbol", "")).strip().upper()
                sector = str(row.get("Sector", "")).strip()
                company_name = str(row.get("Company Name", "")).strip()
                listed_shares = str(row.get("Listed Shares", "")).strip()
                paidup_value = str(row.get("Paidup Value", "")).strip()
                total_paidup = str(row.get("Total Paidup", "")).strip()
                
                if not symbol:
                    excluded_rows += 1
                    continue
                    
                if not is_equity(symbol=symbol, sector=sector, name=company_name):
                    excluded_rows += 1
                    continue
                    
                kept_rows += 1
                
                # Double check to prevent unique constraint failures
                existing = db.query(Company).filter(Company.symbol == symbol).first()
                if not existing:
                    company = Company(
                        symbol=symbol,
                        sector=sector,
                        company_name=company_name,
                        listed_shares=listed_shares,
                        paidup_value=paidup_value,
                        total_paidup=total_paidup,
                        instrument_type="equity"
                    )
                    db.add(company)
                    
                    if len(first_10) < 10:
                        first_10.append(symbol)
                        
            db.commit()
            
            logger.info("--- Company Seeding Summary ---")
            logger.info(f"Total CSV rows read: {total_rows}")
            logger.info(f"Rows kept as equities: {kept_rows}")
            logger.info(f"Rows excluded: {excluded_rows}")
            logger.info(f"First 10 inserted symbols: {first_10}")
            logger.info("-------------------------------")
            
    except Exception as e:
        logger.error(f"STARTUP ERROR: Failed to seed companies safely. Please ensure mapping is correct in the CSV. Error: {e}")
        db.rollback()
        raise e  # Visible startup error so it doesn't silently continue

def get_company_by_ticker(db: Session, ticker: str):
    return db.query(Company).filter(Company.symbol == ticker.upper()).first()

def get_all_companies(db: Session):
    return db.query(Company).all()

def search_companies(db: Session, query: str):
    search = f"%{query}%"
    return db.query(Company).filter(
        (Company.symbol.ilike(search)) | (Company.company_name.ilike(search))
    ).all()
