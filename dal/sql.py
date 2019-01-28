from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Sequence, Integer, Float, String, DateTime, func

from config import Config

Base = declarative_base()


engine = create_engine(
    'postgresql://usr:pass@localhost:5432/sqlalchemy', echo=True)


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, Sequence('upload_id_seq'), primary_key=True)
    process_key = Column(String, unique=True, index=True)
    received_on = Column(DateTime, index=True, default=func.now())
    client_ip = Column(String)
    nb_pictures = Column(Integer)


class DB():

    def __init__(self, uri=None, details=None):
        self._session = self._create_session(uri, **details)

    def _create_session(self, uri, **details):
        engine = create_engine(uri, echo=True)
        Base.metada.create_all(engine)
        return sessionmaker(bind=engine)()

    def list_uploads(self):
        return self._session.query(Upload).order_by(Upload.received_on.asc()).all()

    def create_upload(self, key, dte, ip, nb_pics):
        self._session.add(Upload(
            process_key=key,
            client_ip=ip,
            nb_pictures=nb_pics
        ))
        self._session.commit()
