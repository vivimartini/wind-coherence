#!/usr/bin/env python3
"""Deploy the live scoring service.

Synthesise from this directory. Credentials are needed to deploy, not to synth.
Node has to be on PATH, and on this cluster the jsii cache must sit in a
private directory because the home cache is group-writable:

    export PATH=/rds/user/vm481/hpc-work/wind-coherence/node/bin:$PATH
    export JSII_RUNTIME_PACKAGE_CACHE_ROOT=/rds/user/vm481/hpc-work/wind-coherence/jsii-cache
    export GBWIND_ALERT_EMAIL=you@example.com
    /rds/user/vm481/hpc-work/wind-coherence/venv-cdk/bin/python app.py

The tasks run in us-west-2, the same region as the dynamical.org ensemble
buckets, with no NAT gateway. The container image is built and pushed to the
repository this stack creates. Until that image exists, the task definitions
are in the account and the schedules are the thing to inspect.
"""

import os
from pathlib import Path

import aws_cdk as cdk

from stack import LiveStack

app = cdk.App(outdir=str(Path(__file__).resolve().parent / "cdk.out"))
LiveStack(
    app,
    "GbwindLive",
    env=cdk.Environment(region="us-west-2"),
    alert_email=app.node.try_get_context("alert_email") or os.environ.get("GBWIND_ALERT_EMAIL", ""),
)
app.synth()
