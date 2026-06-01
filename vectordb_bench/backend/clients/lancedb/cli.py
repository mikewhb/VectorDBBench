from datetime import date
from pathlib import Path
from typing import Annotated, Unpack

import json

import click
from pydantic import SecretStr

from .... import config
from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..api import IndexType, MetricType


def _parse_storage_options(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        opts = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"--storage-options-json is not valid JSON: {exc}"
        raise click.BadParameter(msg) from exc
    if not isinstance(opts, dict):
        msg = "--storage-options-json must decode to a JSON object"
        raise click.BadParameter(msg)
    return opts


def _parse_metric_type(raw: str | None) -> MetricType:
    return MetricType[raw.upper()] if raw else MetricType.COSINE


def _parse_int_csv(raw: str | None, default: int) -> list[int]:
    if raw is None or not raw.strip():
        return [default]

    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            msg = f"Expected comma-separated integers, got {raw!r}"
            raise click.BadParameter(msg) from exc

    return values or [default]


def _build_db_config(parameters: dict):
    from .config import LanceDBConfig

    return LanceDBConfig(
        db_label=parameters["db_label"],
        uri=parameters["uri"],
        token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        storage_options=_parse_storage_options(parameters.get("storage_options_json")),
    )


class LanceDBTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="URI connection string", required=True),
    ]
    token: Annotated[
        str | None,
        click.option("--token", type=str, help="Authentication token", required=False),
    ]
    storage_options_json: Annotated[
        str | None,
        click.option(
            "--storage-options-json",
            type=str,
            required=False,
            help=(
                "JSON string forwarded as object_store storage_options to lancedb.connect(). "
                "Useful for S3-compatible backends (e.g. Tencent COS)."
            ),
        ),
    ]


class LanceDBMetricTypedDict(LanceDBTypedDict):
    metric_type: Annotated[
        str | None,
        click.option(
            "--metric-type",
            type=click.Choice(["L2", "COSINE", "IP", "DP"], case_sensitive=False),
            default="COSINE",
            show_default=True,
            help="Distance metric used to build and query the Lance index",
        ),
    ]


class LanceDBCommonIVFTypedDict(LanceDBMetricTypedDict):
    num_partitions: Annotated[
        int,
        click.option(
            "--num-partitions",
            type=int,
            default=0,
            help="Number of IVF partitions, 0 = LanceDB default",
        ),
    ]
    sample_rate: Annotated[
        int,
        click.option(
            "--sample-rate",
            type=int,
            default=256,
            show_default=True,
            help="Training sample rate",
        ),
    ]
    max_iterations: Annotated[
        int,
        click.option(
            "--max-iterations",
            type=int,
            default=50,
            show_default=True,
            help="KMeans max iterations for IVF training",
        ),
    ]
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes",
            type=int,
            default=0,
            help="Search nprobes, 0 = LanceDB default",
        ),
    ]
    refine_factor: Annotated[
        int,
        click.option(
            "--refine-factor",
            type=int,
            default=0,
            help=(
                "Rerank approximate candidates with raw vectors; 0 = unset. "
                "Larger values may improve recall but usually reduce QPS."
            ),
        ),
    ]
    curve_nprobes: Annotated[
        str | None,
        click.option(
            "--curve-nprobes",
            type=str,
            default=None,
            help=(
                "Comma-separated nprobes values for a recall/QPS curve. "
                "When set, the command runs one benchmark point per value."
            ),
        ),
    ]
    curve_refine_factors: Annotated[
        str | None,
        click.option(
            "--curve-refine-factors",
            type=str,
            default=None,
            help=(
                "Comma-separated refine_factor values for a recall/QPS curve. "
                "Use with --curve-nprobes to scan the recall/QPS tradeoff."
            ),
        ),
    ]
    curve_summary_path: Annotated[
        str | None,
        click.option(
            "--curve-summary-path",
            type=str,
            default=None,
            help=(
                "TSV path for recall/QPS curve points. Defaults to "
                "vectordb_bench/results/LanceDB/curve_<date>_<task_label>.tsv when curve mode is enabled."
            ),
        ),
    ]


def _iter_curve_parameters(parameters: dict):
    nprobes_values = _parse_int_csv(parameters.get("curve_nprobes"), parameters["nprobes"])
    refine_values = _parse_int_csv(parameters.get("curve_refine_factors"), parameters["refine_factor"])
    is_curve = parameters.get("curve_nprobes") is not None or parameters.get("curve_refine_factors") is not None

    if not is_curve:
        yield parameters
        return

    base_task_label = parameters["task_label"]
    base_db_label = parameters["db_label"]
    point_idx = 0
    for nprobes in nprobes_values:
        for refine_factor in refine_values:
            point = dict(parameters)
            point["nprobes"] = nprobes
            point["refine_factor"] = refine_factor
            suffix = f"np{nprobes}_rf{refine_factor}"
            point["task_label"] = f"{base_task_label}_{suffix}"
            point["db_label"] = f"{base_db_label}_{suffix}"

            # Curve points after the first reuse the data and index.  This keeps
            # the curve focused on search-time recall/QPS knobs such as rerank.
            if point_idx > 0:
                point["drop_old"] = False
                point["load"] = False
                point["rebuild_index"] = False
            point_idx += 1
            yield point


def _curve_enabled(parameters: dict) -> bool:
    return parameters.get("curve_nprobes") is not None or parameters.get("curve_refine_factors") is not None


def _curve_summary_path(parameters: dict) -> Path:
    raw_path = parameters.get("curve_summary_path")
    if raw_path:
        return Path(raw_path)
    file_name = f"curve_{date.today().strftime('%Y%m%d')}_{parameters['task_label']}.tsv"
    return config.RESULTS_LOCAL_DIR.joinpath("LanceDB", file_name)


def _append_curve_summary(parameters: dict, summary_path: Path) -> None:
    result_path = config.RESULTS_LOCAL_DIR.joinpath(
        "LanceDB",
        f"result_{date.today().strftime('%Y%m%d')}_{parameters['task_label']}_lancedb.json",
    )
    if not result_path.exists():
        click.echo(f"Curve summary skipped, result file not found: {result_path}", err=True)
        return

    result = json.loads(result_path.read_text())
    case_result = result["results"][0]
    metrics = case_result["metrics"]
    db_case_config = case_result["task_config"]["db_case_config"]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a") as f:
        if write_header:
            f.write(
                "\t".join(
                    [
                        "task_label",
                        "db_label",
                        "index_type",
                        "num_partitions",
                        "nprobes",
                        "refine_factor",
                        "recall",
                        "ndcg",
                        "qps",
                        "serial_latency_p95",
                        "serial_latency_p99",
                        "conc_num_list",
                        "conc_qps_list",
                        "conc_latency_p95_list",
                        "result_file",
                    ]
                )
                + "\n"
            )
        f.write(
            "\t".join(
                [
                    parameters["task_label"],
                    parameters["db_label"],
                    str(db_case_config.get("index", "")),
                    str(db_case_config.get("num_partitions", "")),
                    str(db_case_config.get("nprobes", "")),
                    str(db_case_config.get("refine_factor", "")),
                    str(metrics.get("recall", "")),
                    str(metrics.get("ndcg", "")),
                    str(metrics.get("qps", "")),
                    str(metrics.get("serial_latency_p95", "")),
                    str(metrics.get("serial_latency_p99", "")),
                    ",".join(map(str, metrics.get("conc_num_list", []))),
                    ",".join(map(str, metrics.get("conc_qps_list", []))),
                    ",".join(map(str, metrics.get("conc_latency_p95_list", []))),
                    str(result_path),
                ]
            )
            + "\n"
        )


def _run_lancedb_curve(parameters: dict, make_case_config):
    curve_mode = _curve_enabled(parameters)
    summary_path = _curve_summary_path(parameters) if curve_mode else None
    for point in _iter_curve_parameters(parameters):
        run(
            db=DB.LanceDB,
            db_config=_build_db_config(point),
            db_case_config=make_case_config(point),
            **point,
        )
        if curve_mode and not point["dry_run"]:
            _append_curve_summary(point, summary_path)


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDB(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBNoIndexConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(parameters),
        db_case_config=LanceDBNoIndexConfig(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDBAutoIndex(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBAutoIndexConfig

    run(
        db=DB.LanceDB,
        db_config=_build_db_config(parameters),
        db_case_config=LanceDBAutoIndexConfig(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBCommonIVFTypedDict)
def LanceDBIVFFlat(**parameters: Unpack[LanceDBCommonIVFTypedDict]):
    from .config import LanceDBIVFFlatConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFFlatConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBCommonIVFTypedDict)
def LanceDBIVFSQ(**parameters: Unpack[LanceDBCommonIVFTypedDict]):
    from .config import LanceDBIVFSQConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFSQConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )


class LanceDBIVFPQTypedDict(LanceDBCommonIVFTypedDict):
    num_sub_vectors: Annotated[
        int,
        click.option(
            "--num-sub-vectors",
            type=int,
            default=0,
            help="Number of sub-vectors for IVF_PQ, 0 = LanceDB default",
        ),
    ]
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=int,
            default=8,
            show_default=True,
            help="Bits per sub-vector for IVF_PQ (4 or 8)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFPQTypedDict)
def LanceDBIVFPQ(**parameters: Unpack[LanceDBIVFPQTypedDict]):
    from .config import LanceDBIVFPQConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFPQConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            num_sub_vectors=p["num_sub_vectors"],
            nbits=p["nbits"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )


class LanceDBIVFRQTypedDict(LanceDBCommonIVFTypedDict):
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=int,
            default=1,
            show_default=True,
            help="Bits per dimension for IVF_RQ",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFRQTypedDict)
def LanceDBIVFRQ(**parameters: Unpack[LanceDBIVFRQTypedDict]):
    from .config import LanceDBIVFRQConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFRQConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            nbits=p["nbits"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )


class LanceDBIVFHNSWSQTypedDict(LanceDBCommonIVFTypedDict):
    m: Annotated[
        int,
        click.option("--m", type=int, default=20, show_default=True, help="HNSW m parameter"),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            default=300,
            show_default=True,
            help="HNSW ef_construction parameter",
        ),
    ]
    ef: Annotated[
        int,
        click.option("--ef", type=int, default=0, help="Search ef, 0 = unset"),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFHNSWSQTypedDict)
def LanceDBIVFHNSWFlat(**parameters: Unpack[LanceDBIVFHNSWSQTypedDict]):
    from .config import LanceDBIVFHNSWFlatConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFHNSWFlatConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            m=p["m"],
            ef_construction=p["ef_construction"],
            ef=p["ef"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFHNSWSQTypedDict)
def LanceDBIVFHNSWSQ(**parameters: Unpack[LanceDBIVFHNSWSQTypedDict]):
    from .config import LanceDBIVFHNSWSQConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFHNSWSQConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            m=p["m"],
            ef_construction=p["ef_construction"],
            ef=p["ef"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )


class LanceDBIVFHNSWPQTypedDict(LanceDBIVFHNSWSQTypedDict):
    num_sub_vectors: Annotated[
        int,
        click.option(
            "--num-sub-vectors",
            type=int,
            default=0,
            help="Number of sub-vectors for IVF_HNSW_PQ, 0 = LanceDB default",
        ),
    ]
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=int,
            default=8,
            show_default=True,
            help="Bits per sub-vector for IVF_HNSW_PQ (4 or 8)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFHNSWPQTypedDict)
def LanceDBIVFHNSWPQ(**parameters: Unpack[LanceDBIVFHNSWPQTypedDict]):
    from .config import LanceDBIVFHNSWPQConfig

    _run_lancedb_curve(
        parameters,
        lambda p: LanceDBIVFHNSWPQConfig(
            metric_type=_parse_metric_type(p.get("metric_type")),
            num_partitions=p["num_partitions"],
            sample_rate=p["sample_rate"],
            max_iterations=p["max_iterations"],
            m=p["m"],
            ef_construction=p["ef_construction"],
            ef=p["ef"],
            num_sub_vectors=p["num_sub_vectors"],
            nbits=p["nbits"],
            nprobes=p["nprobes"],
            refine_factor=p["refine_factor"],
        ),
    )
