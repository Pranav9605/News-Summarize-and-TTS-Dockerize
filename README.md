# News Summarization & Text-to-Speech Application

The application extracts key details from multiple news articles for a specified company, performs sentiment and comparative analyses, and generates a Hindi text-to-speech (TTS) output. The frontend is built with Streamlit and communicates with a FastAPI backend via REST APIs.

## Key Features
- **News Extraction:** Scrapes at least 10 news articles using BeautifulSoup.
- **Sentiment Analysis:** Classifies articles as Positive, Negative, or Neutral.
- **Comparative Analysis:** Aggregates sentiment data and highlights coverage differences.
- **Text-to-Speech:** Converts the final sentiment summary into Hindi audio.
- **User Interface:** A Streamlit-based UI allowing company selection/input.
- **API Integration:** RESTful endpoints handle communication between the frontend and backend.

## Project Structure
```
News-Summarize-and-TTS-Dockerize/
├── app.py              # Streamlit frontend
├── api.py              # FastAPI backend endpoints
├── utils.py            # Utility functions for news extraction, analysis, and TTS
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration (for local testing/deployment)
└── README.md           # Project documentation
```

## Setup & Deployment

### Local Development

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Pranav9605/News-Summarize-and-TTS-Dockerize.git
   cd News-Summarize-and-TTS-Dockerize
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate    # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Backend:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   Access the API documentation at [https://pranav9605-test2.hf.space/docs](https://pranav9605-test2.hf.space/docs).

5. **Run the Frontend:**
   Open a separate terminal with the virtual environment activated:
   ```bash
   streamlit run app.py
   ```
   The Streamlit app will launch in your default browser.

### Deployment on Hugging Face Spaces

- The frontend is deployed as a standalone Streamlit application.
- **Deployment URL:** [https://news-summarize-and-tts-application.streamlit.app/](https://news-summarize-and-tts-application.streamlit.app/)

## API Endpoints

- **POST /api/news**  
  *Description:* Fetch news articles for a company.  
  *Payload Example:*
  ```json
  { "company_name": "Tesla" }
  ```

- **POST /api/sentiment**  
  *Description:* Generate a full sentiment analysis report from the news articles.  
  *Payload Example:*
  ```json
  { "company_name": "Tesla" }
  ```

- **POST /api/tts**  
  *Description:* Convert text to Hindi speech.  
  *Query Parameters:*  
    - `text`: Text to convert  
    - `lang`: Language code (default: "hi")

## Model Details

- **News Extraction:** Uses BeautifulSoup to scrape article data.
- **Sentiment Analysis:** Implements sentiment analysis (e.g., VADER or TextBlob) to classify article content.
- **Comparative Analysis:** Aggregates sentiment metrics and highlights differences.
- **Text-to-Speech:** Utilizes an open-source TTS model for Hindi audio output.

## Assumptions & Limitations

- **Data Sources:** Assumes non-JS weblinks can be scraped reliably.
- **TTS Quality:** Dependent on the underlying TTS model.
- **Error Handling:** Basic error handling is implemented; further robustness may be added as needed.
- **API Communication:** Frontend and backend interact via REST APIs with CORS enabled.

## Submission Details

- **GitHub Repository:** [https://github.com/Pranav9605/News-Summarize-and-TTS-Dockerize](https://github.com/Pranav9605/News-Summarize-and-TTS-Dockerize)
- **Hugging Face Spaces Deployment:** [https://huggingface.co/spaces/Pranav9605/test2/tree/main](https://huggingface.co/spaces/Pranav9605/test2/tree/main)
- **Video Demo:** A video demonstration has been submitted through the provided form.

## Contact

For further inquiries, please contact:  
**Email:** pranavpadmanabhan1234@gmail.com
