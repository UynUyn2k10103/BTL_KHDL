from models.predict_one_sentence import predict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Sentence(BaseModel):
    sentence : str

    
@app.post("/sentence")
async def upload_sentence(sentence: Sentence):
    # get label
    label = predict(sentence.sentence)

    json_compatible_item_data = jsonable_encoder(label)

    # save in database
    
    
    return JSONResponse(json_compatible_item_data)
    # return label