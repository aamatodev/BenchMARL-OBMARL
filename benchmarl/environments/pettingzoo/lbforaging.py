#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    players: int = MISSING
    min_player_level: int = MISSING
    max_player_level: int = MISSING
    min_food_level: int = MISSING
    max_food_level: int = MISSING
    field_size: tuple[int] = MISSING
    max_num_food: int = MISSING
    sight: int = MISSING
    max_cycles: int = MISSING
    force_coop: int = MISSING
