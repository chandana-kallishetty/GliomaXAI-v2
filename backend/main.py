from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import predict, upload, report, analytics, cases, auth

app = FastAPI(
    title="GliomaXAI API",
    description="Brain MRI Classification Backend",
    version="1.0.0",
)

# -- CORS ---------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Global fallback exception handler ----------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[global] Unhandled {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {str(exc)}"},
    )


# -- Routers -------------------------------------------------------------------
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(upload.router,  tags=["Upload"])
app.include_router(predict.router, tags=["Predict"])
app.include_router(report.router,  tags=["Report"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(cases.router, prefix="/api", tags=["Cases"])


# -- Root / health endpoints ---------------------------------------------------
@app.get("/", tags=["Health"])
def home():
    return {"status": "online", "message": "GliomaXAI Backend Running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}