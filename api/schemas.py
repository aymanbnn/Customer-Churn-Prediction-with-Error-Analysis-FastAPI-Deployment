from pydantic import BaseModel
from typing import Dict

class CustomerData(BaseModel):
    features: Dict[str, float]