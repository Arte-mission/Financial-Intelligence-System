# Nepal Stock Market API

A small FastAPI backend for fetching market news sentiment and stock statistics for the Nepal stock market.

## Tech Stack
- **Python**
- **FastAPI**
- **SQLite & SQLAlchemy**: For basic storage structuring.
- **Requests & BeautifulSoup4**: For ShareSansar web scraping.
- **TextBlob**: For sentiment analysis.
- **Uvicorn**: ASGI web server.
- **Playwright**: Included as an available dependency for fallback dynamic setups.

## Project Structure
```text
.
├── .env.example
├── requirements.txt
├── README.md
└── app/
    ├── main.py
    ├── database.py
    ├── models.py
    ├── schemas.py
    ├── services/
    │   ├── newsdata_service.py
    │   ├── sharesansar_service.py
    │   └── sentiment_service.py
    ├── routers/
    │   ├── market.py
    │   └── stock.py
    └── utils/
        └── config.py
```

## Setup Instructions

1. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   - Copy the `.env.example` file to create a `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Obtain an API key from [NewsData.io](https://newsdata.io/) and paste it into `.env`:
     ```env
     NEWSDATA_API_KEY=your_real_key_here
     ```

4. **Run the Application:**
   ```bash
   uvicorn app.main:app --reload
   ```
   The backend API will start securely at `http://127.0.0.1:8000`.

## API Endpoints

- **GET `/market-mood`**: Retrieves the latest Nepal business news, performs sentiment analysis using TextBlob on the headlines, and computes a market mood label (`Bullish`, `Neutral`, `Bearish`) along with an average sentiment score (-1.0 to 1.0).
- **GET `/stock/{ticker}`**: Extracts the latest stock details such as `current_price`, `price_change_percent`, and the top news headlines for the specified company ticker by scraping information from sharesansar.com.

## API Documentation
Once running, you can access the automatically generated interactive API documentation (Swagger UI) by navigating to:
`http://127.0.0.1:8000/docs`
