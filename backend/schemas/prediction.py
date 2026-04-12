from typing import List

from pydantic import BaseModel


class PredictConfig(BaseModel):
    datos: List[List[float]]
    normalizar: bool = False
