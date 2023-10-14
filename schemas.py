from pydantic import BaseModel

class CreateUserRequest(BaseModel):
    surname : str
    name : str
    role : str

