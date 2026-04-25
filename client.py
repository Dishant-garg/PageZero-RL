# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PageZero Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import PageZeroAction, PageZeroObservation, PageZeroState
except (ImportError, ValueError):
    from models import PageZeroAction, PageZeroObservation, PageZeroState


class PageZeroEnvClient(
    EnvClient[PageZeroAction, PageZeroObservation, PageZeroState]
):
    """
    Client for the PageZero Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    """

    def _step_payload(self, action: PageZeroAction) -> Dict:
        """
        Convert PageZeroAction to JSON payload for step message.
        """
        return {
            "tool": action.tool,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PageZeroObservation]:
        """
        Parse server response into StepResult[PageZeroObservation].
        """
        obs_data = payload.get("observation")
        if obs_data is None:
            # Fallback: payload itself is the observation
            obs_data = payload
        
        # Ensure reward/done are in obs_data for the model constructor
        obs_data = dict(obs_data)
        if "reward" in payload:
            # Coerce None to 0.0 to satisfy Pydantic float type
            val = payload.get("reward")
            obs_data["reward"] = float(val) if val is not None else 0.0
        
        if "done" in payload:
            # Coerce None to False to satisfy Pydantic bool type
            val = payload.get("done")
            obs_data["done"] = bool(val) if val is not None else False
        
        observation = PageZeroObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> PageZeroState:
        """
        Parse server response into State object.
        """
        return PageZeroState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", 0.15),
            scenario_name=payload.get("scenario_name", "None"),
            is_resolved=payload.get("is_resolved", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
