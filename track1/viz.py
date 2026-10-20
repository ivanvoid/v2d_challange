"""Small drawing helpers shared by the review sheets (numpy only, works in every venv)."""
from __future__ import annotations

import numpy as np


def contrast_bg(colors) -> tuple[int, int, int]:
    """A background colour that stands out against an object whose pixel/vertex colours are `colors`.

    The opposite of the object's mean colour; if that is about as bright as the object (a mid-grey object has a
    mid-grey opposite), it is pushed to light or dark instead, so a dark pan never sits on a dark background.
    colors: (..., 3) in 0-1 or 0-255.
    """
    c = np.asarray(colors, float).reshape(-1, 3)
    if c.size == 0:
        return (128, 128, 128)
    if c.max() <= 1.0:
        c = c * 255
    mean = c.mean(0)
    bg = 255 - mean
    lum = lambda x: float(x @ [0.299, 0.587, 0.114])
    if abs(lum(bg) - lum(mean)) < 90:
        bg = np.array([235.0, 235, 235]) if lum(mean) < 128 else np.array([25.0, 25, 25])
    return tuple(int(round(v)) for v in np.clip(bg, 0, 255))
