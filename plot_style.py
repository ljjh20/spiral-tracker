from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


_CONFIGURED = False


def _register_fonts(font_dir: Path) -> None:
    if not font_dir.exists():
        return

    patterns = (
        "lmroman*.otf",
        "lmsans*.otf",
        "lmmono*.otf",
        "latinmodern-*.otf",
    )
    for pattern in patterns:
        for path in sorted(font_dir.glob(pattern)):
            try:
                font_manager.fontManager.addfont(str(path))
            except (OSError, RuntimeError, ValueError):
                continue


def _first_available(candidates: list[str], available: set[str], default: str) -> str:
    for name in candidates:
        if name in available:
            return name
    return default


def configure_matplotlib_defaults() -> None:
    """Use Latin Modern fonts for plots when available, with sane serif fallbacks."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    for font_dir in (
        Path.home() / "Library/Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    ):
        _register_fonts(font_dir)

    available = {font.name for font in font_manager.fontManager.ttflist}

    serif = _first_available(
        ["LMRoman12", "LMRoman10", "LMRoman17", "Latin Modern Roman", "Computer Modern Roman", "CMU Serif"],
        available,
        "DejaVu Serif",
    )
    sans = _first_available(
        ["LMSans10", "LMSans12", "Latin Modern Sans"],
        available,
        "DejaVu Sans",
    )
    mono = _first_available(
        ["LMMono10", "LMMono12", "Latin Modern Mono"],
        available,
        "DejaVu Sans Mono",
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(
                dict.fromkeys(
                    [
                        serif,
                        "LMRoman12",
                        "LMRoman10",
                        "Latin Modern Roman",
                        "Computer Modern Roman",
                        "CMU Serif",
                        "DejaVu Serif",
                    ]
                )
            ),
            "font.sans-serif": [
                sans,
                "LMSans10",
                "Latin Modern Sans",
                "DejaVu Sans",
            ],
            "font.monospace": [
                mono,
                "LMMono10",
                "Latin Modern Mono",
                "DejaVu Sans Mono",
            ],
            "mathtext.fontset": "custom",
            "mathtext.rm": serif,
            "mathtext.it": f"{serif}:italic",
            "mathtext.bf": f"{serif}:bold",
            "mathtext.sf": sans,
            "mathtext.tt": mono,
        }
    )

    _CONFIGURED = True
