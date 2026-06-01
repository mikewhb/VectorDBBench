from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class LanceDBConfig(DBConfig):
    """LanceDB connection configuration."""

    db_label: str
    uri: str
    token: SecretStr | None = None
    # Optional object_store storage options (for example Tencent COS).
    storage_options: dict | None = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "token": self.token.get_secret_value() if self.token else None,
            "storage_options": self.storage_options,
        }


class LanceDBCommonIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType
    metric_type: MetricType = MetricType.L2
    num_partitions: int = 0
    sample_rate: int = 256
    max_iterations: int = 50
    target_partition_size: int = 0
    nprobes: int = 0
    refine_factor: int = 0

    def parse_metric(self) -> str:
        if self.metric_type in [MetricType.L2, MetricType.COSINE]:
            return self.metric_type.value.lower()
        if self.metric_type in [MetricType.IP, MetricType.DP]:
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported for LanceDB!"
        raise ValueError(msg)

    def common_index_param(self) -> dict:
        params = {
            "metric": self.parse_metric(),
            "sample_rate": self.sample_rate,
            "max_iterations": self.max_iterations,
        }
        if self.num_partitions > 0:
            params["num_partitions"] = self.num_partitions
        if self.target_partition_size > 0:
            params["target_partition_size"] = self.target_partition_size
        return params

    def search_param(self) -> dict:
        params = {}
        if self.nprobes > 0:
            params["nprobes"] = self.nprobes
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params


LanceDBIndexConfig = LanceDBCommonIndexConfig


class LanceDBNoIndexConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.NONE

    def index_param(self) -> dict:
        return {}


class LanceDBAutoIndexConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {}


class LanceDBIVFFlatConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_FLAT"
        return params


class LanceDBIVFSQConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.IVF_SQ

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_SQ"
        return params


class LanceDBIVFPQConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.IVFPQ
    num_sub_vectors: int = 0
    nbits: int = 8

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_PQ"
        params["num_bits"] = self.nbits
        if self.num_sub_vectors > 0:
            params["num_sub_vectors"] = self.num_sub_vectors
        return params


class LanceDBIVFRQConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.IVF_RQ
    nbits: int = 1

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_RQ"
        params["num_bits"] = self.nbits
        return params


class LanceDBIVFHNSWFlatConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.IVF_HNSW_FLAT
    m: int = 20
    ef_construction: int = 300
    ef: int = 0

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_HNSW_FLAT"
        params["m"] = self.m
        params["ef_construction"] = self.ef_construction
        return params

    def search_param(self) -> dict:
        params = super().search_param()
        if self.ef > 0:
            params["ef"] = self.ef
        return params


class LanceDBIVFHNSWSQConfig(LanceDBCommonIndexConfig):
    index: IndexType = IndexType.IVF_HNSW_SQ
    m: int = 20
    ef_construction: int = 300
    ef: int = 0

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_HNSW_SQ"
        params["m"] = self.m
        params["ef_construction"] = self.ef_construction
        return params

    def search_param(self) -> dict:
        params = super().search_param()
        if self.ef > 0:
            params["ef"] = self.ef
        return params


class LanceDBIVFHNSWPQConfig(LanceDBIVFHNSWSQConfig):
    index: IndexType = IndexType.IVF_HNSW_PQ
    num_sub_vectors: int = 0
    nbits: int = 8

    def index_param(self) -> dict:
        params = self.common_index_param()
        params["index_type"] = "IVF_HNSW_PQ"
        params["m"] = self.m
        params["ef_construction"] = self.ef_construction
        params["num_bits"] = self.nbits
        if self.num_sub_vectors > 0:
            params["num_sub_vectors"] = self.num_sub_vectors
        return params


_lancedb_case_config = {
    IndexType.NONE: LanceDBNoIndexConfig,
    IndexType.AUTOINDEX: LanceDBAutoIndexConfig,
    IndexType.IVFFlat: LanceDBIVFFlatConfig,
    IndexType.IVF_SQ: LanceDBIVFSQConfig,
    IndexType.IVFPQ: LanceDBIVFPQConfig,
    IndexType.IVF_RQ: LanceDBIVFRQConfig,
    IndexType.IVF_HNSW_FLAT: LanceDBIVFHNSWFlatConfig,
    IndexType.IVF_HNSW_SQ: LanceDBIVFHNSWSQConfig,
    IndexType.IVF_HNSW_PQ: LanceDBIVFHNSWPQConfig,
}
