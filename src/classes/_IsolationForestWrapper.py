from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Any

# Proposta de algoritmo alternativo - Isolation Forest
# 1.1 Princípio: anomalias são pontos raros e facilmente isolados em poucas
# divisões de árvores binárias. 1.2 Funcionamento: um conjunto de “arbóres de
# isolamento” é treinado de forma aleatória e a profundidade para isolar uma
# instância define seu score de anomalia. 1.3 Vantagens: complexidade O(n log n)
# e independência de rótulos. 1.4 Implementação: utilizamos a classe
# IsolationForest da biblioteca scikit-learn.


class IsolationForestWrapper:
    """Wrapper para compatibilizar IsolationForest com a interface utilizada na
    validação do pipeline."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = IsolationForest(**kwargs)

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Retorna ação equivalente à detecção de anomalia."""
        obs_2d = np.atleast_2d(obs)
        pred = self.model.predict(obs_2d)
        action = np.where(pred == -1, 1, 0)
        return action[0], None