"""Run final-project training locally.

Usage:
    uv run python scripts/train.py [training arguments]
    uv run python scripts/train.py reward_model [training arguments]
    uv run python scripts/train.py rm_grpo [training arguments]
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable


TRAINERS: dict[str, str] = {
    "policy": "llm_rl_final_proj.train",
    "reward_model": "llm_rl_final_proj.reward_model.train",
    "rm_grpo": "llm_rl_final_proj.online.train_rm_grpo",
}


def _select_trainer(argv: list[str]) -> tuple[str, list[str]]:
    if argv and argv[0] in TRAINERS:
        return argv[0], argv[1:]
    return "policy", argv


def main() -> None:
    trainer_name, trainer_args = _select_trainer(sys.argv[1:])
    os.environ.setdefault("REQUIRE_CUDA", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    module_name = TRAINERS[trainer_name]
    module = __import__(module_name, fromlist=["main"])
    train_main: Callable[[], None] = module.main
    sys.argv = [f"{trainer_name}_train.py", *trainer_args]
    train_main()


if __name__ == "__main__":
    main()