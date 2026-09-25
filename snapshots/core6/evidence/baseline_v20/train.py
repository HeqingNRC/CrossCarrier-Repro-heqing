from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / "EXPERIMENTSRESULT"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from v15_v18_common_train import main


if __name__ == "__main__":
    main()
