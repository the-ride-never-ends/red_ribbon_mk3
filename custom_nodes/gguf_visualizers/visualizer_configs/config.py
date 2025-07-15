import os
import logging
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field

from ._get_config import get_config as config

logger = logging.getLogger(__name__)

# Define hard-coded constants
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = _THIS_DIR = os.path.dirname(_THIS_DIR)
MAIN_FOLDER = os.path.dirname(_THIS_DIR)
YEAR_IN_DAYS: int = 365
DEBUG_FILEPATH: str = os.path.join(MAIN_FOLDER, "debug_logs")
RANDOM_SEED: int = 420

# Define program-specific hard-coded constants
# Clip values to at max 7 standard deviations from the mean.
CFG_SD_CLIP_THRESHOLD = 7
# Number of standard deviations above the mean to be positively scaled.
CFG_SD_POSITIVE_THRESHOLD = 1
# Number of standard deviations below the mean to be negatively scaled.
CFG_SD_NEGATIVE_THRESHOLD = 1
# RGB scaling for pixels that meet the negative threshold.
CFG_NEG_SCALE = (1.2, 0.2, 1.2)
# RGB scaling for pixels that meet the positive threshold.
CFG_POS_SCALE = (0.2, 1.2, 1.2)
# RGB scaling for pixels between those ranges.
CFG_MID_SCALE = (0.1, 0.1, 0.1)
# CFG_MID_SCALE = (0.6, 0.6, 0.9) Original Values



class GGUFVisualizerConfig(BaseModel):
    
    SKIP_STEPS: bool = Field(default=True)
    FILENAME_PREFIX: str = Field(default=f"{MAIN_FOLDER}")
    INPUT_FILENAME: str = Field(default="input.csv")
    INPUT_FOLDER: str = Field(default_factory=os.path.join(MAIN_FOLDER, "input"))
    OUTPUT_FOLDER: str = Field(default_factory=os.path.join(MAIN_FOLDER, "output"))
    MAIN_FOLDER: str = Field(default=MAIN_FOLDER)

    @property
    def PROJECT_ROOT(self) -> str:
        return os.path.dirname(_THIS_DIR)

    @property
    def DEBUG_FILEPATH(self) -> str:
        return os.path.join(self.MAIN_FOLDER, "debug_logs")

    @property
    def DAYS_IN_TEAR(self) -> Literal[365]:
        return 365

    @property
    def RANDOM_SEED(self) -> Literal[420]:
        return 420

