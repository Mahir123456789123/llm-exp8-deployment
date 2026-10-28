# Experiment 7 — Customer Support RAG API (Render)

FastAPI deployment for the Experiment 7 customer-support RAG pipeline.

## Endpoints
- `GET /` — service information
- `GET /health` — health check
- `POST /ask` — ask a customer-support question
- `GET /docs` — interactive Swagger API docs

## Deploy on Render
1. Upload these files to a GitHub repository.
2. In Render, choose **New + → Blueprint** and connect the repository containing `render.yaml`, or create a **Web Service** from the repository.
3. For a Blueprint, Render reads `render.yaml`. For a manual Web Service, use:
   - Runtime: Python
   - Build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. In Render's Environment settings, add `HF_TOKEN` as a secret environment variable. It must be a Hugging Face token authorized to make calls to Inference Providers. Never commit the token to GitHub.
5. Confirm `HF_MODEL` is currently available through Hugging Face Inference Providers for your account. If not, set it to a model that is available to your account.
6. Deploy and check `/health`, `/docs`, and then submit a POST request to `/ask`.

## Example request
```json
{
  "question": "How do I request a refund?"
}
```

## Notes
- Render's free instances have limited memory and may spin down when idle. Loading the embedding model and FAISS index can use substantial RAM; if startup fails or the service is terminated, inspect the Render logs for memory errors.
- The app uses Hugging Face Inference Providers for text generation. API availability and usage limits depend on the selected model and account.
- Do not commit API tokens, passwords, or other secrets.
