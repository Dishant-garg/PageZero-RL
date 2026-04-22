# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PageZero environment server components.

Lazy imports — PageZeroEnvironment is only loaded when the OpenEnv server
starts (requires openenv-core). StackBackend and friends can be imported
standalone without the openenv package.
"""

# Do NOT eagerly import PageZeroEnvironment here.
# It requires openenv.core which is only available in the full server env.
# Import it explicitly inside server/app.py instead.

__all__ = ["PageZeroEnvironment"]


def __getattr__(name: str):
    if name == "PageZeroEnvironment":
        from .PageZero_environment import PageZeroEnvironment  # noqa: PLC0415
        return PageZeroEnvironment
    raise AttributeError(f"module 'server' has no attribute {name!r}")
