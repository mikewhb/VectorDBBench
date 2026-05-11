"""CLI commands for TDSQL Multimodal client.

Each index type gets its own subcommand so users can pick the index by
selecting a different command (parallels Milvus / AliSQL conventions).
"""

import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class TDSQLMultimodalTypedDict(CommonTypedDict):
    """Common TDSQL Multimodal connection options."""

    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="TDSQL Multimodal host (e.g. 127.0.0.1)",
            required=True,
        ),
    ]

    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="TDSQL Multimodal port (no default; must match the running instance)",
            required=True,
        ),
    ]

    user_name: Annotated[
        str,
        click.option(
            "--user",
            "user_name",
            type=str,
            help="MySQL user name",
            default="root",
            show_default=True,
        ),
    ]

    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="MySQL password (defaults to env $TDSQL_MULTIMODAL_PASSWORD or empty)",
            default=lambda: os.environ.get("TDSQL_MULTIMODAL_PASSWORD", ""),
            show_default=False,
        ),
    ]

    db_name: Annotated[
        str,
        click.option(
            "--db-name",
            type=str,
            help="Optional default database (ignored: client always uses lance.main)",
            default="",
            show_default=False,
        ),
    ]


class TDSQLMultimodalIVFFlatTypedDict(TDSQLMultimodalTypedDict):
    metric_type: Annotated[
        str | None,
        click.option(
            "--metric-type",
            type=click.Choice(["L2", "COSINE", "IP"], case_sensitive=False),
            help="Distance metric. Defaults to COSINE (matches OpenAI/Cohere datasets).",
            default="COSINE",
            show_default=True,
        ),
    ]

    num_partitions: Annotated[
        int | None,
        click.option(
            "--num-partitions",
            type=int,
            help="IVF_FLAT num_partitions (default: heuristic ~ sqrt(N))",
            required=False,
        ),
    ]

    nprobs: Annotated[
        int | None,
        click.option(
            "--nprobs",
            type=int,
            help="IVF_FLAT search nprobs",
            required=False,
        ),
    ]

    refine_factor: Annotated[
        int | None,
        click.option(
            "--refine-factor",
            type=int,
            help="IVF_FLAT search refine_factor",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalIVFFlatTypedDict)
def TDSQLMultimodalIVFFlat(  # noqa: N802 - command name matches CLI convention
    **parameters: Unpack[TDSQLMultimodalIVFFlatTypedDict],
):
    from ..api import MetricType
    from .config import TDSQLMultimodalConfig, TDSQLMultimodalIVFFlatConfig

    # Map CLI string -> framework MetricType enum.
    metric_type = MetricType[parameters["metric_type"].upper()] if parameters.get("metric_type") else MetricType.COSINE

    run(
        db=DB.TDSQLMultimodal,
        db_config=TDSQLMultimodalConfig(
            db_label=parameters["db_label"],
            user_name=parameters["user_name"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=TDSQLMultimodalIVFFlatConfig(
            metric_type=metric_type,
            num_partitions=parameters["num_partitions"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )
