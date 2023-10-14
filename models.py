from sqlalchemy import Integer, String, Float, Boolean
from sqlalchemy.sql.schema import Column
from database import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    surname = Column(String)
    name = Column(String)
    role = Column(String)
    token = Column(String)
