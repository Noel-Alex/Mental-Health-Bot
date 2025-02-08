from pydantic import BaseModel

class UserBase(BaseModel):
    userEid: str
    ageApprox: int
    profession: str

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True
