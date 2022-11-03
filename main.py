from typing import Union
from models.predict_one_sentence import predict

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.param_functions import Depends

app = FastAPI()

# class Sentence(BaseModel):
#     sentence : str

# @app.get("/")
# def read_root():
#     return {'Hello' : 'World'}


    
@app.post("/sentence/")

async def upload_sentence(sentence : str = Body(..., embed=True)):
    # get label
    label = await predict(sentence)
    print(label)

    json_compatible_item_data = jsonable_encoder(label)

    # save in database
    
    
    return JSONResponse(json_compatible_item_data)
    
