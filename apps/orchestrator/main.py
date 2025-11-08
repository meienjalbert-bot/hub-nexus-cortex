from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from core.memory.mome_router import route as mome_route
from core.llm.multi_llm_voting import vote as llm_vote
from core.llm.model_manager import prewarm as models_prewarm
from core.orchestration.predictive_scheduler import predict_plan

app = FastAPI(title="hub-nexus-cortex")

class VoteBody(BaseModel):
    prompt: str
    context: str = ""
    experts: list[str] | None = None

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/route")
async def route(q: str = Query(..., min_length=1), k: int = 5):
    return await mome_route(q, k)

@app.post("/vote")
async def vote(body: VoteBody):
    return await llm_vote(body.prompt, body.context or "")

@app.get("/schedule/predict")
def schedule_predict():
    return predict_plan()

@app.post("/models/swap")
async def models_swap(payload: dict = Body(...)):
    models = payload.get("prewarm", [])
    ok = await models_prewarm(models)
    return {"ok": ok, "models": models}
