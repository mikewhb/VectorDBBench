"""CLI commands for TDSQL Multimodal client.

Each index type gets its own subcommand so users can pick the index by
selecting a different command.
"""

import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class TDSQLMultimodalTypedDict(CommonTypedDict):
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
            help="TDSQL Multimodal port (must match the running instance)",
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


class TDSQLMultimodalMetricTypedDict(TDSQLMultimodalTypedDict):
    metric_type: Annotated[
        str | None,
        click.option(
            "--metric-type",
            type=click.Choice(["L2", "COSINE", "IP", "DP"], case_sensitive=False),
            help="Distance metric. Defaults to COSINE.",
            default="COSINE",
            show_default=True,
        ),
    ]


def _parse_metric_type(raw: str | None) -> MetricType:
    return MetricType[raw.upper()] if raw else MetricType.COSINE


def _build_db_config(parameters: dict):
    from .config import TDSQLMultimodalConfig

    return TDSQLMultimodalConfig(
        db_label=parameters["db_label"],
        user_name=parameters["user_name"],
        password=SecretStr(parameters["password"]),
        host=parameters["host"],
        port=parameters["port"],
        db_name=parameters["db_name"],
    )


class TDSQLMultimodalCommonIVFTypedDict(TDSQLMultimodalMetricTypedDict):
    num_partitions: Annotated[
        int | None,
        click.option(
            "--num-partitions",
            type=int,
            help="IVF num_partitions (default: heuristic ~ sqrt(N))",
            required=False,
        ),
    ]
    nprobs: Annotated[
        int | None,
        click.option(
            "--nprobs",
            type=int,
            help="Search nprobs",
            required=False,
        ),
    ]
    refine_factor: Annotated[
        int | None,
        click.option(
            "--refine-factor",
            type=int,
            help="Search refine_factor",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalCommonIVFTypedDict)
def TDSQLMultimodalIVFFlat(**parameters: Unpack[TDSQLMultimodalCommonIVFTypedDict]):
    from .config import TDSQLMultimodalIVFFlatConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFFlatConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class TDSQLMultimodalIVFPQTypedDict(TDSQLMultimodalCommonIVFTypedDict):
    num_sub_vectors: Annotated[
        int | None,
        click.option(
            "--num-sub-vectors",
            type=int,
            help="IVF_PQ num_sub_vectors",
            required=False,
        ),
    ]
    num_bits: Annotated[
        int | None,
        click.option(
            "--num-bits",
            type=int,
            help="IVF_PQ num_bits",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalIVFPQTypedDict)
def TDSQLMultimodalIVFPQ(**parameters: Unpack[TDSQLMultimodalIVFPQTypedDict]):
    from .config import TDSQLMultimodalIVFPQConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFPQConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            num_sub_vectors=parameters["num_sub_vectors"],
            num_bits=parameters["num_bits"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalCommonIVFTypedDict)
def TDSQLMultimodalIVFSQ(**parameters: Unpack[TDSQLMultimodalCommonIVFTypedDict]):
    from .config import TDSQLMultimodalIVFSQConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFSQConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class TDSQLMultimodalIVFRQTypedDict(TDSQLMultimodalCommonIVFTypedDict):
    num_bits: Annotated[
        int | None,
        click.option(
            "--num-bits",
            type=int,
            help="IVF_RQ num_bits",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalIVFRQTypedDict)
def TDSQLMultimodalIVFRQ(**parameters: Unpack[TDSQLMultimodalIVFRQTypedDict]):
    from .config import TDSQLMultimodalIVFRQConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFRQConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            num_bits=parameters["num_bits"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class TDSQLMultimodalIVFHNSWSQTypedDict(TDSQLMultimodalCommonIVFTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="IVF_HNSW_SQ HNSW m",
            required=False,
        ),
    ]
    ef_construction: Annotated[
        int | None,
        click.option(
            "--ef-construction",
            type=int,
            help="IVF_HNSW_SQ HNSW ef_construction",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalIVFHNSWSQTypedDict)
def TDSQLMultimodalIVFHNSWFlat(**parameters: Unpack[TDSQLMultimodalIVFHNSWSQTypedDict]):
    from .config import TDSQLMultimodalIVFHNSWFlatConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFHNSWFlatConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalIVFHNSWSQTypedDict)
def TDSQLMultimodalIVFHNSWSQ(**parameters: Unpack[TDSQLMultimodalIVFHNSWSQTypedDict]):
    from .config import TDSQLMultimodalIVFHNSWSQConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFHNSWSQConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class TDSQLMultimodalIVFHNSWPQTypedDict(TDSQLMultimodalIVFHNSWSQTypedDict):
    num_sub_vectors: Annotated[
        int | None,
        click.option(
            "--num-sub-vectors",
            type=int,
            help="IVF_HNSW_PQ num_sub_vectors",
            required=False,
        ),
    ]
    num_bits: Annotated[
        int | None,
        click.option(
            "--num-bits",
            type=int,
            help="IVF_HNSW_PQ num_bits",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TDSQLMultimodalIVFHNSWPQTypedDict)
def TDSQLMultimodalIVFHNSWPQ(**parameters: Unpack[TDSQLMultimodalIVFHNSWPQTypedDict]):
    from .config import TDSQLMultimodalIVFHNSWPQConfig

    run(
        db=DB.TDSQLMultimodal,
        db_config=_build_db_config(parameters),
        db_case_config=TDSQLMultimodalIVFHNSWPQConfig(
            metric_type=_parse_metric_type(parameters.get("metric_type")),
            num_partitions=parameters["num_partitions"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            num_sub_vectors=parameters["num_sub_vectors"],
            num_bits=parameters["num_bits"],
            nprobs=parameters["nprobs"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )
