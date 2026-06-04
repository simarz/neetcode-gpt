import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sbt = z - np.max(z)
        expZ = np.exp(sbt)
        res = expZ/np.sum(expZ)
        return np.round(res, 4)