from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bot import query_rag_engine
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def handle_chat(request: Request):
    try:
        payload = await request.json()
        user_message = payload.get("message", "")
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message is required"}
            )
        
        # Get response from RAG engine
        response = query_rag_engine(user_message)
        
        return JSONResponse(
            content={
                "response": response
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)