from typing import List

from pydantic import BaseModel


class LayerConfig(BaseModel):
    neuronas: int = 32
    activacion: str = "ReLU"


class TrainingConfig(BaseModel):
    k: int = 5
    eta: float = 0.01
    eps: float = 0.1
    capas_config: List[LayerConfig] = [LayerConfig(neuronas=64), LayerConfig(neuronas=32)]
    neuronas_salida: int = 1
    activacion_salida: str = "Lineal"
    max_epochs: int = 300
    normalizar: bool = True
    paciencia: int = 0   # 0 = desactivado; N = épocas sin mejora en error total antes de parar
