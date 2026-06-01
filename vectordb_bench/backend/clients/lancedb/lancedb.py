import logging
import os
import random
import time
from contextlib import contextmanager

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel

from ..api import IndexType, VectorDB
from .config import LanceDBConfig, LanceDBIndexConfig

log = logging.getLogger(__name__)

# Process-wide shared Session so all connections in this process share the same
# index/metadata cache. Built lazily once we read the env-driven cache size.
_SHARED_SESSION = None


def _get_shared_session():
    """Build (or fetch) a process-wide lancedb.Session honoring env
    LANCEDB_INDEX_CACHE_BYTES (>0 enables it). Returns None when unset so the
    legacy default-session path is preserved."""
    global _SHARED_SESSION
    cache_bytes_env = os.environ.get("LANCEDB_INDEX_CACHE_BYTES", "").strip()
    if not cache_bytes_env:
        return None
    try:
        cache_bytes = int(cache_bytes_env)
    except ValueError:
        log.warning(f"Invalid LANCEDB_INDEX_CACHE_BYTES={cache_bytes_env!r}, ignored")
        return None
    if cache_bytes <= 0:
        return None
    if _SHARED_SESSION is None:
        try:
            from lancedb import Session
            _SHARED_SESSION = Session(index_cache_size_bytes=cache_bytes)
            log.info(f"Created shared lancedb Session with index_cache_size_bytes={cache_bytes}")
        except Exception as e:
            log.warning(f"Failed to create lancedb Session (cache disabled): {e}")
            return None
    return _SHARED_SESSION


class VectorModel(LanceModel):
    id: int
    vector: list[float]


class LanceDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: LanceDBConfig,
        db_case_config: LanceDBIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "LanceDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = os.environ.get("LANCEDB_TABLE_NAME", "").strip() or collection_name
        self.table_location = os.environ.get("LANCEDB_TABLE_LOCATION", "").strip() or None
        self.vector_column = os.environ.get("LANCEDB_VECTOR_COLUMN", "").strip() or "vector"
        self.dim = dim
        self.uri = db_config["uri"]
        # Optional object_store storage options (e.g. s3-compat for Tencent COS).
        self.storage_options = db_config.get("storage_options")
        # avoid the search_param being called every time during the search process
        self.search_config = db_case_config.search_param()

        log.info(f"Search config: {self.search_config}")
        if self.storage_options:
            # Avoid logging credentials.
            log.info(f"Storage options keys: {list(self.storage_options.keys())}")

        db = self._connect()

        if self.table_location:
            log.info(
                f"Opening Lance table by explicit location: name={self.table_name}, "
                f"location={self.table_location}, vector_column={self.vector_column}"
            )
            self._open_table(db)
            return

        if drop_old:
            try:
                db.drop_table(self.table_name)
            except Exception as e:
                log.warning(f"Failed to drop table {self.table_name}: {e}")

        try:
            db.open_table(self.table_name)
        except Exception:
            schema = pa.schema(
                [pa.field("id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), list_size=self.dim))]
            )
            db.create_table(self.table_name, schema=schema, mode="overwrite")

    def _connect(self):
        # Pass storage_options only when set; keeps local-fs path unchanged.
        kwargs = {}
        if self.storage_options:
            kwargs["storage_options"] = self.storage_options
        session = _get_shared_session()
        if session is not None:
            kwargs["session"] = session
        return lancedb.connect(self.uri, **kwargs)

    @contextmanager
    def init(self):
        self.db = self._connect()
        self.table = self._open_table(self.db)
        try:
            if os.environ.get("LANCEDB_PREWARM", "").strip() == "1":
                self._prewarm_indices()
            yield
        finally:
            self.db = None
            self.table = None

    def _open_table(self, db):
        if not self.table_location:
            return db.open_table(self.table_name)

        from lancedb.table import LanceTable

        return LanceTable.open(
            db,
            self.table_name,
            location=self.table_location,
            storage_options=self.storage_options,
        )

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            data = [{ "id": meta, self.vector_column: emb} for meta, emb in zip(metadata, embeddings, strict=False)]
            self.table.add(data)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into LanceDB table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        def search(query):
            try:
                return self.table.search(query, vector_column_name=self.vector_column)
            except TypeError:
                return self.table.search(query)

        if filters:
            results = (
                search(query)
                .select(["id", "_distance"])
                .where(f"id >= {filters['id']}", prefilter=True)
                .limit(k)
            )
        else:
            results = search(query).select(["id", "_distance"]).limit(k)

        if self.search_config.get("nprobes", 0) > 0:
            results = results.nprobes(self.search_config["nprobes"])
        if self.search_config.get("refine_factor", 0) > 0:
            results = results.refine_factor(self.search_config["refine_factor"])
        if self.search_config.get("ef", 0) > 0:
            results = results.ef(self.search_config["ef"])

        return [int(result["id"]) for result in results.to_list()]

    def optimize(self, data_size: int | None = None):
        if not (self.table and hasattr(self, "case_config") and self.case_config.index != IndexType.NONE):
            return

        params = self.case_config.index_param()
        log.info(f"Creating index for LanceDB table ({self.table_name})")
        log.info(f"Index parameters: {params}")

        self._drop_existing_indices()
        self.table.create_index(**params)
        if self.case_config.index in (
            IndexType.IVFFlat,
            IndexType.IVF_SQ,
            IndexType.IVFPQ,
            IndexType.IVF_RQ,
            IndexType.IVF_HNSW_FLAT,
            IndexType.IVF_HNSW_PQ,
            IndexType.AUTOINDEX,
        ):
            self.table.optimize()

        # Optional dual-prewarm:
        #   1) lancedb prewarm_index (loads index into Session index cache)
        #   2) probe search with random vectors (fills OS page cache + lazy paths)
        if os.environ.get("LANCEDB_PREWARM", "").strip() == "1":
            self._prewarm_indices()

    def _drop_existing_indices(self) -> None:
        try:
            indices = list(self.table.list_indices())
        except Exception as e:
            log.warning(f"list_indices failed before rebuild, skip index drop: {e}")
            return

        for idx in indices:
            name = getattr(idx, "name", None)
            if not name:
                continue
            try:
                log.info(f"Dropping LanceDB index {name} on table {self.table_name}")
                self.table.drop_index(name)
            except Exception as e:
                log.warning(f"Failed to drop LanceDB index {name}: {e}")

    def _prewarm_indices(self) -> None:
        """Two-stage warm-up: lancedb prewarm_index per index, then random search probes."""
        try:
            indices = list(self.table.list_indices())
        except Exception as e:
            log.warning(f"list_indices failed, skip prewarm: {e}")
            return
        log.info(f"Prewarm: discovered {len(indices)} indices: {[getattr(i, 'name', i) for i in indices]}")
        for idx in indices:
            name = getattr(idx, "name", None)
            if not name:
                continue
            t0 = time.perf_counter()
            try:
                self.table.prewarm_index(name)
                log.info(f"prewarm_index({name}) done in {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                log.warning(f"prewarm_index({name}) failed: {e}")

        # Stage 2: 100 random-vector probes to fill OS page cache + any lazy paths.
        try:
            n_probes = int(os.environ.get("LANCEDB_PREWARM_PROBES", "100"))
        except ValueError:
            n_probes = 100
        if n_probes <= 0:
            return
        rnd = random.Random(42)
        t0 = time.perf_counter()
        for _ in range(n_probes):
            vec = [rnd.random() for _ in range(self.dim)]
            try:
                try:
                    query = self.table.search(vec, vector_column_name=self.vector_column)
                except TypeError:
                    query = self.table.search(vec)
                query.select(["id"]).limit(10).to_list()
            except Exception as e:
                log.warning(f"prewarm probe search failed: {e}")
                break
        log.info(f"Prewarm probe: {n_probes} searches done in {time.perf_counter() - t0:.2f}s")
