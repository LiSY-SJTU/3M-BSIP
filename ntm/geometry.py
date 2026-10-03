from typing import Tuple
import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算将模板点集 P 对齐到目标点集 Q 的刚体变换 (R, t)，最小化 RMSD。
    P, Q: (N,3)
    返回: R(3,3), t(3,)
    """
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P and Q must be of shape (N,3)")

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be of shape (N,3)")
    return (R @ coords.T).T + t


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    diff = A - B
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
