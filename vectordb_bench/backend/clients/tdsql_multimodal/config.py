"""Config for TDSQL Multimodal client.

TDSQL Multimodal exposes a Lance catalog over the MySQL wire protocol.
All vector tables live under the fixed catalog/schema ``lance.main``.
Tables created by this client are prefixed with ``vdb_`` for namespace
isolation, and the index type is encoded as a suffix in the table name
(e.g. ``vdb_cohere_medium_1m_ivfflat``).
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
        if self.metric_type == MetricType.IP:
            # Lance accepts 'dot' for inner product / dot product.
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported"
        raise ValueError(msg)


class TDSQLMultimodalIVFFlatConfig(TDSQLMultimodalIndexConfig, DBCaseConfig):
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

    num_partitions: int | None = None
    nprobs: int | None = None
    refine_factor: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
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


_tdsql_multimodal_case_config = {
    IndexType.IVFFlat: TDSQLMultimodalIVFFlatConfig,
}
