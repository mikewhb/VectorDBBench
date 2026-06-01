"""Config for TDSQL Multimodal client.

TDSQL Multimodal exposes a Lance catalog over the MySQL wire protocol.
All vector tables live under the fixed catalog/schema ``lance.main``.
Tables created by this client are prefixed with ``vdb_`` for namespace
isolation, and the concrete SQL index type is encoded as a suffix in the
table name (for example ``vdb_cohere_medium_1m_ivfflat`` or
``vdb_cohere_medium_1m_ivfhnswsq``).
"""

from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


# Fixed Lance catalog/schema. Lance ships ``main`` as the default schema and
# we never USE another database; all access goes through fully-qualified names.
LANCE_CATALOG = "lance"
LANCE_SCHEMA = "main"
TABLE_PREFIX = "vdb_"


class TDSQLMultimodalConfigDict(TypedDict):
    """Connection kwargs for ``mysql.connector.connect``."""

    host: str
    port: int
    user: str
    password: str


class TDSQLMultimodalConfig(DBConfig):
    """Connection config for TDSQL Multimodal (Lance catalog over MySQL wire).

    Notes:
      * ``host`` / ``port`` are required (no defaults) - users must point at
        the exact instance to avoid accidental writes.
      * ``db_name`` is accepted for CLI symmetry but ignored by the client:
        Lance catalog already defaults to ``lance.main``, and we use fully
        qualified table names everywhere.
    """

    user_name: str = "root"
    password: SecretStr = SecretStr("")
    host: str
    port: int
    # Accepted for CLI symmetry; the client never issues USE.
    db_name: str = ""

    # Allow empty password by default (root with empty pwd is the common dev setup).
    _extra_empty_skip = frozenset({"password", "db_name"})

    def to_dict(self) -> TDSQLMultimodalConfigDict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": self.password.get_secret_value(),
        }


class TDSQLMultimodalIndexConfig(BaseModel):
    """Base index config; subclasses pin a concrete index type."""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        """Map VectorDBBench MetricType to Lance ``metric_type`` literal."""
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type in {MetricType.IP, MetricType.DP}:
            # Lance accepts 'dot' for inner product / dot product.
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported"
        raise ValueError(msg)




class _TDSQLMultimodalIVFBaseConfig(TDSQLMultimodalIndexConfig, DBCaseConfig):
    num_partitions: int | None = None
    nprobs: int | None = None
    refine_factor: int | None = None
    index: IndexType

    def base_index_param(self) -> dict:
        return {
            "index_type": self.index.value,
            "metric_type": self.parse_metric(),
            "num_partitions": self.num_partitions,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "nprobs": self.nprobs,
            "refine_factor": self.refine_factor,
        }


class TDSQLMultimodalIVFFlatConfig(_TDSQLMultimodalIVFBaseConfig):
    """IVF_FLAT index config.

    Index DDL (per Lance MTR baseline):
      CREATE INDEX <name> ON lance.main.<table> (vec)
        USING IVF_FLAT WITH (num_partitions=N, metric_type='l2');

    Search TVF:
      SELECT id FROM lance_vector_search(
          'lance.main.<table>', 'vec', <query>::FLOAT[D],
          k = K, nprobs = P, refine_factor = R)
      ORDER BY _distance ASC, id;
    """

    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return self.base_index_param()


class TDSQLMultimodalIVFPQConfig(_TDSQLMultimodalIVFBaseConfig):
    """IVF_PQ index config.

    Typical DDL:
      CREATE INDEX <name> ON lance.main.<table> (vec)
        USING IVF_PQ WITH (
          num_partitions=N,
          num_sub_vectors=M,
          num_bits=B,
          metric_type='cosine'
        );

    Search still goes through ``lance_vector_search(..., nprobs=..., refine_factor=...)``.
    """

    num_sub_vectors: int | None = None
    num_bits: int | None = None
    index: IndexType = IndexType.IVFPQ

    def index_param(self) -> dict:
        params = self.base_index_param()
        params["num_sub_vectors"] = self.num_sub_vectors
        params["num_bits"] = self.num_bits
        return params


class TDSQLMultimodalIVFHNSWSQConfig(_TDSQLMultimodalIVFBaseConfig):
    """IVF_HNSW_SQ index config."""

    m: int | None = None
    ef_construction: int | None = None
    index: IndexType = IndexType.IVF_HNSW_SQ

    def index_param(self) -> dict:
        params = self.base_index_param()
        # SQLEngine's Lance FFI expects HNSW build parameters with the hnsw_
        # prefix. Passing LanceDB-style names such as "m" / "ef_construction"
        # is accepted by SQL parsing but ignored by the Rust index builder.
        params["hnsw_m"] = self.m
        params["hnsw_ef_construction"] = self.ef_construction
        return params


class TDSQLMultimodalIVFSQConfig(_TDSQLMultimodalIVFBaseConfig):
    """IVF_SQ index config."""

    index: IndexType = IndexType.IVF_SQ

    def index_param(self) -> dict:
        return self.base_index_param()


class TDSQLMultimodalIVFRQConfig(_TDSQLMultimodalIVFBaseConfig):
    """IVF_RQ index config."""

    num_bits: int | None = None
    index: IndexType = IndexType.IVF_RQ

    def index_param(self) -> dict:
        params = self.base_index_param()
        params["num_bits"] = self.num_bits
        return params


class TDSQLMultimodalIVFHNSWFlatConfig(_TDSQLMultimodalIVFBaseConfig):
    """IVF_HNSW_FLAT index config."""

    m: int | None = None
    ef_construction: int | None = None
    index: IndexType = IndexType.IVF_HNSW_FLAT

    def index_param(self) -> dict:
        params = self.base_index_param()
        params["hnsw_m"] = self.m
        params["hnsw_ef_construction"] = self.ef_construction
        return params


class TDSQLMultimodalIVFHNSWPQConfig(TDSQLMultimodalIVFHNSWFlatConfig):
    """IVF_HNSW_PQ index config."""

    num_sub_vectors: int | None = None
    num_bits: int | None = None
    index: IndexType = IndexType.IVF_HNSW_PQ

    def index_param(self) -> dict:
        params = super().index_param()
        params["num_sub_vectors"] = self.num_sub_vectors
        params["num_bits"] = self.num_bits
        return params


_tdsql_multimodal_case_config = {
    IndexType.IVFFlat: TDSQLMultimodalIVFFlatConfig,
    IndexType.IVFPQ: TDSQLMultimodalIVFPQConfig,
    IndexType.IVF_HNSW_FLAT: TDSQLMultimodalIVFHNSWFlatConfig,
    IndexType.IVF_HNSW_SQ: TDSQLMultimodalIVFHNSWSQConfig,
    IndexType.IVF_HNSW_PQ: TDSQLMultimodalIVFHNSWPQConfig,
    IndexType.IVF_SQ: TDSQLMultimodalIVFSQConfig,
    IndexType.IVF_RQ: TDSQLMultimodalIVFRQConfig,
}
