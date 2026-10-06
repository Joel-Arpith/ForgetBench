from .retrain import run_retrain, run_sisa
from .npo_rmu import run_npo, run_rmu
from .hru import run_hru

# "from_scratch": trains a fresh model that never sees the forget set.
# "scrub": starts from the pristine (forget-baked-in) model and edits it.
REGISTRY = {
    "retrain": ("from_scratch", run_retrain),
    "sisa": ("from_scratch", run_sisa),
    "npo": ("scrub", run_npo),
    "rmu": ("scrub", run_rmu),
    "hru": ("scrub", run_hru),
}
