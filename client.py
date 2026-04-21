# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PageZero Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PageZeroAction, PageZeroObservation


class PageZeroEnvClient(
    EnvClient[PageZeroAction, PageZeroObservation, State]
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
        obs_data = payload.get("observation", {})
        observation = PageZeroObservation(
            tool_output=obs_data.get("tool_output", ""),
            active_alerts=obs_data.get("active_alerts", []),
            sla_status=obs_data.get("sla_status", "OK"),
            revenue_loss_usd=obs_data.get("revenue_loss_usd", 0.0),
            downtime_minutes=obs_data.get("downtime_minutes", 0.0),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 15),
            hint=obs_data.get("hint", None),
            phase_history=obs_data.get("phase_history", []),
            is_done=obs_data.get("is_done", False),
            final_score=obs_data.get("final_score", None),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
