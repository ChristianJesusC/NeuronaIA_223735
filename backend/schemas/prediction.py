from typing import List

from pydantic import BaseModel


class PredictConfig(BaseModel):
    datos: List[str]   # líneas CSV crudas, igual que en el archivo original
    normalizar: bool = False
