from pydantic import BaseModel 

class attack_description(BaseModel):
    timestamp:str 
    method:str
    status:int
    size:int
    url:str



