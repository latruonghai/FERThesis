from sqlalchemy import Column, String, Integer, Text
from .. import Base


class Tourism(Base):
    
    __tablename__ = "Tourism"
    ids = Column(Integer, primary_key=True)
    keyword = Column(String(200), nullable=False)
    title = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(300), nullable=False)

if __name__ == "__main__":
    pass