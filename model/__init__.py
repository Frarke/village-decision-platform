"""村庄地质灾害综合整治决策模型计算内核。"""

from .solver import SolverConfig, solve, solve_exact
from .nsga2 import solve_nsga2

__all__ = ["SolverConfig", "solve", "solve_exact", "solve_nsga2"]
