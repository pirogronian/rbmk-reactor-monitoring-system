from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from redis_client import r


app = FastAPI()

# CORS(Cross-Origin Resource Sharing)
app.add_middleware( # By default, internet browsers and servers block the requests if scripts tries to send data to API that works under different address
    CORSMiddleware, # CORS allows these requests to reach the API
    allow_origins=["*"], # This means allows request from any URL address
    allow_methods=["*"], # This means approval to use any HTTP method (GET, POST etc.)
    allow_headers=["*"], # This means approval to send any headers (Content-Type: application/json)
)

# BaseModel enabled automatic parsing, validation and generating documentation
class GrafanaData(BaseModel):
    value: int | None = None




@app.on_event("startup")
def startup():
    '''Delete leftover data on startup'''

    r.delete("rods_movement")
    r.delete("water_flow")
    r.delete("in_water_temp")
    r.delete("az5")

@app.post("/prety")
async def grafana_webhook_rods(data: GrafanaData):
    '''Take data about rod movement from Grafana'''

    print(f"Otrzymano wartość przez POST pręty: {data.value}", flush=True)
    if data.value is None:
        return("Value is None.")
    r.lpush("rods_movement", data.value)
    # return int(value)


@app.post("/water")
async def grafana_webhook_water(data: GrafanaData):
    '''Take data about water flow from Grafana'''

    print(f"Otrzymano wartość przez POST water: {data.value}", flush=True)
    if data.value is None:
        return("Value is None.")
    r.lpush("water_flow", data.value)


@app.post("/in_water_temp")
async def grafana_webhook_water(data: GrafanaData):
    '''Take data about inlet water temperature from Grafana'''

    print(f"Otrzymano wartość przez POST in_water_temp: {data.value}", flush=True)
    if data.value is None:
        return("Value is None.")
    r.lpush("in_water_temp", data.value)


@app.post("/az5")
async def grafana_webhook_az5(data: GrafanaData):
    '''Take information about AZ5 procedure initialization'''

    print(f"Otrzymano sygnał AZ5 {data.value}")
    if data.value is None:
        return ("Value is None.")
    r.lpush("az5", data.value)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
