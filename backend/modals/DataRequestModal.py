from pydantic import BaseModel
from typing import Dict, List

class DataReqModal(BaseModel):
    pages : Dict[str, List[int]]