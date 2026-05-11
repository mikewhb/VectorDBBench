"""TDSQL Multimodal VectorDB client.

Connects to TDSQL Multimodal (SQLEngine) via the MySQL wire protocol and
operates on Lance-backed tables under the fixed ``lance.main`` schema.

Key SQL shapes (verified against MTR baseline ``lance_index_ops.test`` and
``lance_vector_search.test``):

  CREATE TABLE lance.main.<tbl> (id BIGINT, vec FLOAT[<dim>]);
  INSERT  INTO lance.main.<tbl> VALUES (?, [..]::FLOAT[<dim>]);
  CREATE INDEX <ix> ON lance.main.<tbl> (vec)
      USING IVF_FLAT WITH (num_partitions=N, metric_type='l2');
  SELECT id FROM lance_vector_search(
      'lance.main.<tbl>', 'vec', [..]::FLOAT[<dim>], k=K, nprobs=P, refine_factor=R)
      ORDER BY _distance ASC, id;
"""

import logging
import re
import threading
from contextlib import contextmanager

import mysql.connector as mysql

from ..api import VectorDB
from .config import (
    LANCE_CATALOG,
    LANCE_SCHEMA,
    TABLE_PREFIX,
    TDSQLMultimodalConfigDict,
    TDSQLMultimodalIndexConfig,
)

log = logging.getLogger(__name__)


# Suffix mapping: VectorDBBench IndexType.value -> table-name suffix.
# Keep concise (no underscores) so table names stay short and SQL-safe.
_INDEX_SUFFIX = {
    "IVF_FLAT": "ivfflat",
    "IVF_PQ": "ivfpq",
    "HNSW": "hnsw",
}


def _sanitize_table_name(collection_name: str, index_type_value: str) -> str:
    """Build the physical table name from the case collection name and index type.

    The function is idempotent w.r.t. ``TABLE_PREFIX``: if the caller already
    passes a name that starts with ``vdb_`` we do not prepend the prefix
    again. This keeps table names short and avoids repeats like
    ``vdb_vdb_default_ivfflat``.

    Example:
      collection_name='cohere-medium-1m', index_type_value='IVF_FLAT'
      -> 'vdb_cohere_medium_1m_ivfflat'
      collection_name='vdb_perf50k', index_type_value='IVF_FLAT'
      -> 'vdb_perf50k_ivfflat'
    """
    base = re.sub(r"[^A-Za-z0-9]+", "_", collection_name).strip("_").lower()
    suffix = _INDEX_SUFFIX.get(index_type_value, index_type_value.lower().replace("_", ""))
    if base.startswith(TABLE_PREFIX):
        return f"{base}_{suffix}"
    return f"{TABLE_PREFIX}{base}_{suffix}"


def _vec_to_literal(v) -> str:  # noqa: ANN001
    """Render a Python float sequence as a Lance vector literal.

    Output form: ``[0.1,0.2,...]`` (without the ``::FLOAT[N]`` cast suffix;
    the cast is appended in the SQL template that consumes this literal).

    We render the literal manually rather than relying on driver parameter
    binding because ``mysql.connector`` has no native binding for Lance's
    typed array literal syntax.
    """
    # repr() on float gives a round-trippable representation; join with ','
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class TDSQLMultimodal(VectorDB):
    """VectorDB client for TDSQL Multimodal (Lance catalog over MySQL wire).

    Concurrency model:
      Each worker thread gets its own ``mysql.connector`` connection via
      ``threading.local``. Lance itself supports concurrent appends through
      MVCC + optimistic commit (manifest put-if-not-exists with auto retry).
      Multiple MySQL sessions therefore translate to multiple DuckDB
      connections on the server side, each driving an independent Lance
      append commit. Default concurrency is the framework's
      ``min(cpu_count, 4)``; override via ``--load-concurrency N`` on the CLI.
    """

    # mysql.connector connection objects are not shareable across threads,
    # but we side-step that by giving each worker its own connection (see
    # _get_local()), so the client itself is safe under concurrent
    # insert_embeddings() / search_embedding() calls.
    thread_safe: bool = True

    def __init__(
        self,
        dim: int,
        db_config: TDSQLMultimodalConfigDict,
        db_case_config: TDSQLMultimodalIndexConfig,
        collection_name: str = "default",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "TDSQLMultimodal"
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim

        # Physical table name encodes dataset + index type to keep parallel
        # benchmark runs isolated.
        index_type_value = self.case_config.index_param()["index_type"]
        self.table_name = _sanitize_table_name(collection_name, index_type_value)
        self.full_table = f"{LANCE_CATALOG}.{LANCE_SCHEMA}.{self.table_name}"

        # Thread-local storage is created lazily in init() (which runs inside
        # the spawn-ed subprocess). Storing it here would prevent the runner
        # from pickling the client across the spawn boundary because
        # threading.local objects are not picklable.
        self._tls = None

        # DDL is performed once on a one-shot connection; the worker pool
        # has not been spun up yet at this point.
        conn, cursor = self._open_connection()
        try:
            if drop_old:
                log.info(f"{self.name} drop table: {self.full_table}")
                cursor.execute(f"DROP TABLE IF EXISTS {self.full_table}")
                conn.commit()
            log.info(f"{self.name} create table: {self.full_table} (dim={dim})")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.full_table} "
                f"(id BIGINT, vec FLOAT[{dim}])"
            )
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    def __getstate__(self):
        """Drop the threading.local() before pickling.

        ConcurrentInsertRunner pickles the whole VectorDB across a 'spawn'
        ProcessPoolExecutor boundary. ``threading.local`` instances cannot
        be pickled, but they hold no value at this point anyway (they are
        only populated inside worker threads after ``init()`` runs in the
        target subprocess).
        """
        state = self.__dict__.copy()
        state["_tls"] = None
        return state

    # --------------------------------------------------------------------- #
    # Connection helpers
    # --------------------------------------------------------------------- #
    def _open_connection(self):
        """Open a fresh MySQL-protocol connection. We never set a default
        database - Lance catalog uses ``lance.main`` and all access is fully
        qualified."""
        conn = mysql.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            buffered=True,
        )
        cursor = conn.cursor()
        return conn, cursor

    def _get_local(self):
        """Return (conn, cursor) for the calling thread, opening on demand.

        mysql.connector connections cannot be shared across threads (their
        wire-protocol buffer is single-producer). Each worker thread therefore
        owns a private connection that lives for the duration of the
        ``with self.db.init():`` block.
        """
        if self._tls is None:
            # Should not happen: init() always sets up _tls before any
            # search/insert path runs. Guard defensively in case a caller
            # bypasses the lifecycle.
            self._tls = threading.local()
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn, cursor = self._open_connection()
            self._tls.conn = conn
            self._tls.cursor = cursor
        return self._tls.conn, self._tls.cursor

    def _close_local(self):
        """Best-effort close of the calling thread's connection."""
        if self._tls is None:
            return
        cursor = getattr(self._tls, "cursor", None)
        conn = getattr(self._tls, "conn", None)
        if cursor is not None:
            try:
                cursor.close()
            except Exception:  # noqa: BLE001
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._tls.cursor = None
        self._tls.conn = None

    # --------------------------------------------------------------------- #
    # VectorDB lifecycle
    # --------------------------------------------------------------------- #
    @contextmanager
    def init(self):
        """Per-process / per-thread connection lifecycle.

        With ``thread_safe=True`` the framework keeps a single VectorDB
        instance in the load-subprocess and dispatches it to N worker
        threads. Connections are opened lazily by ``_get_local()`` from
        each worker thread. This is also the first hook that runs in the
        spawn-ed subprocess, so it is the right place to (re)create the
        ``threading.local()`` storage that was nulled out by ``__getstate__``.
        """
        # Recreate thread-local storage in the current process. The instance
        # we receive here may have just been unpickled in a fresh subprocess.
        self._tls = threading.local()
        try:
            yield
        finally:
            self._close_local()
            self._tls = None

    def ready_to_load(self) -> bool:
        return True

    # --------------------------------------------------------------------- #
    # Insert
    # --------------------------------------------------------------------- #
    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Bulk-insert embeddings using a single multi-row VALUES statement.

        Why multi-row VALUES instead of executemany():
          mysql.connector.executemany() rewrites into N separate execute()
          calls (one round-trip per row). For batch sizes in the hundreds
          this dominates wall time. A single multi-row INSERT collapses the
          batch to one round-trip and matches the Lance MTR baseline
          (lance_dml_insert.test Section 2.2 + lance_vector_search.test).

        Vector literal form follows the MTR baseline exactly:
            INSERT INTO ... VALUES
              (1, [0.1, 0.2, ...]::FLOAT[D]),
              (2, [0.3, 0.4, ...]::FLOAT[D]);
        i.e. unquoted array literal + postfix ``::FLOAT[D]`` cast.
        """
        conn, cursor = self._get_local()
        try:
            n = len(metadata)
            # Build one row literal per embedding. Pre-compute the cast
            # suffix once so the inner loop only does string interpolation.
            cast = f"::FLOAT[{self.dim}]"
            rows = [
                f"({int(metadata[i])}, {_vec_to_literal(embeddings[i])}{cast})"
                for i in range(n)
            ]
            sql = (
                f"INSERT INTO {self.full_table} (id, vec) VALUES "
                + ",".join(rows)
            )
            cursor.execute(sql)
            conn.commit()
            return n, None
        except Exception as e:  # noqa: BLE001
            log.warning(f"Insert failed on {self.full_table}: {e}")
            return 0, e

    # --------------------------------------------------------------------- #
    # Index build (called once between load and search)
    # --------------------------------------------------------------------- #
    def optimize(self, data_size: int | None = None) -> None:
        index_param = self.case_config.index_param()
        index_type = index_param["index_type"]
        metric = index_param["metric_type"]

        if index_type == "IVF_FLAT":
            num_partitions = index_param.get("num_partitions") or self._default_num_partitions(data_size)
            ddl = (
                f"CREATE INDEX {self.table_name}_idx "
                f"ON {self.full_table} (vec) USING IVF_FLAT "
                f"WITH (num_partitions={num_partitions}, metric_type='{metric}')"
            )
        else:
            msg = f"Index type {index_type} not supported yet by TDSQLMultimodal client"
            raise NotImplementedError(msg)

        log.info(f"{self.name} build index: {ddl}")
        # optimize() is called between load and search by the framework, on
        # an arbitrary thread. Use this thread's connection.
        conn, cursor = self._get_local()
        try:
            cursor.execute(ddl)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            log.warning(f"Index build failed on {self.full_table}: {e}")
            raise

    @staticmethod
    def _default_num_partitions(data_size: int | None) -> int:
        """Heuristic for IVF num_partitions when user did not pin one.

        Common rule of thumb: ~ sqrt(N). Clamp to >=1 and <=4096 to avoid
        pathological values on very small or very large datasets.
        """
        if not data_size or data_size <= 0:
            return 1
        # Avoid importing math.sqrt to keep this trivial.
        approx = int(data_size ** 0.5)
        return max(1, min(4096, approx))

    # --------------------------------------------------------------------- #
    # Search
    # --------------------------------------------------------------------- #
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,  # noqa: ARG002 - filtering not yet supported
        timeout: int | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> list[int]:
        sp = self.case_config.search_param()

        # lance_vector_search('table', 'col', <query>::FLOAT[D], k=, nprobs=, refine_factor=)
        extra_args = [f"k = {int(k)}"]
        if sp.get("nprobs") is not None:
            extra_args.append(f"nprobs = {int(sp['nprobs'])}")
        if sp.get("refine_factor") is not None:
            extra_args.append(f"refine_factor = {int(sp['refine_factor'])}")

        # Inline the query vector literal so the wire-protocol shape matches
        # the MTR baseline (no string parameter binding for FLOAT[N]).
        sql = (
            f"SELECT id FROM lance_vector_search("
            f"'{self.full_table}', 'vec', "
            f"{_vec_to_literal(query)}::FLOAT[{self.dim}], {', '.join(extra_args)}"
            f") ORDER BY _distance ASC, id"
        )

        conn, cursor = self._get_local()  # noqa: F841 - conn unused here, kept for symmetry
        try:
            cursor.execute(sql)
            return [row[0] for row in cursor.fetchall()]
        except mysql.Error:
            log.exception(f"Search failed on {self.full_table}")
            raise
