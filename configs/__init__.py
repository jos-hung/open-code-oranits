"""Config package initializer.

Exposes `systemcfg` and `config` symbols at the `configs` package level for
convenient import (e.g., `from configs import ddqn_cfg`). This convenience
import uses `import *` and therefore will place many names into the importer
namespace; for clarity import modules directly in larger modules.
"""

from .systemcfg import *
from .config import *