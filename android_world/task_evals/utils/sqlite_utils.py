# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for interacting with SQLite database on an Android device."""

import logging
import os
import shutil
import sqlite3
import subprocess
import time
from typing import Optional, Type
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import file_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FTS3 / FTS4 compatibility shim
# ---------------------------------------------------------------------------
# Conda's SQLite 3.51+ may be compiled *without* FTS3/FTS4 (only FTS5).
# Android app databases (VLC, Joplin, Broccoli …) use FTS3/FTS4 triggers,
# so DELETE / INSERT on regular tables can fail with
#   "sqlite3.OperationalError: no such module: fts3/fts4"
# Workaround: fall back to the system ``sqlite3`` CLI binary which normally
# ships with FTS3/FTS4 enabled.
# ---------------------------------------------------------------------------

def _python_sqlite_has_fts() -> bool:
    """Return True if Python's sqlite3 module supports both FTS3 and FTS4."""
    for fts in ("fts3", "fts4"):
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute(f"CREATE VIRTUAL TABLE _t USING {fts}(c TEXT)")
            conn.close()
        except sqlite3.OperationalError:
            return False
    return True

_HAS_FTS = _python_sqlite_has_fts()
_SYSTEM_SQLITE3: str | None = shutil.which("sqlite3")

if not _HAS_FTS:
    if _SYSTEM_SQLITE3:
        logger.info(
            "Python sqlite3 缺少 FTS3/FTS4，将回退到系统 sqlite3 CLI: %s",
            _SYSTEM_SQLITE3,
        )
    else:
        logger.warning(
            "Python sqlite3 缺少 FTS3/FTS4 且找不到系统 sqlite3 二进制！"
            "涉及 FTS 虚拟表的操作将失败。"
        )


def _checkpoint_wal_on_device(
    remote_db_file_path: str,
    env: interface.AsyncEnv,
    timeout_sec: Optional[float] = None,
) -> bool:
    """在设备上执行 WAL checkpoint，将 WAL 数据合并到主数据库文件。

    使用设备上的 sqlite3 CLI 执行 PRAGMA wal_checkpoint(TRUNCATE)，
    确保所有 WAL 中的数据都被写入主 .db 文件。
    这比 close_app (am force-stop) 更可靠：
    - force-stop 只是杀死进程，并不会触发 WAL checkpoint
    - 设备端 sqlite3 PRAGMA 可以直接完成 checkpoint

    Args:
        remote_db_file_path: 设备上数据库文件的路径。
        env: Android 环境接口。
        timeout_sec: 可选的超时时间。

    Returns:
        True 表示 checkpoint 成功，False 表示失败（设备无 sqlite3 等）。
    """
    try:
        response = adb_utils.issue_generic_request(
            ['shell', 'sqlite3', remote_db_file_path,
             'PRAGMA wal_checkpoint(TRUNCATE);'],
            env.controller,
            timeout_sec,
        )
        # 检查 ADB 请求本身是否成功
        if hasattr(response, 'status'):
            from android_env.proto import adb_pb2
            if response.status != adb_pb2.AdbResponse.Status.OK:
                logger.warning(
                    "设备端 WAL checkpoint 失败 (ADB status=%s): %s",
                    response.status, remote_db_file_path,
                )
                return False
        logger.debug(
            "设备端 WAL checkpoint 成功: %s", remote_db_file_path
        )
        return True
    except Exception as e:
        logger.warning(
            "设备端 WAL checkpoint 异常: %s — %s", remote_db_file_path, e
        )
        return False


def _format_sql_value(value) -> str:
    """Format a Python value as a SQL literal for the sqlite3 CLI."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, bytes):
        return f"X'{value.hex()}'"
    # str — escape single-quotes by doubling them
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _sqlite3_exec(db_path: str, sql: str, timeout: float = 60) -> str:
    """Execute *sql* against *db_path* using the system ``sqlite3`` binary.

    Raises ``sqlite3.OperationalError`` on failure so callers see the same
    exception type as the in-process path.
    """
    if not _SYSTEM_SQLITE3:
        raise sqlite3.OperationalError(
            "系统 sqlite3 二进制不可用，无法绕过 Python FTS 限制"
        )
    result = subprocess.run(
        [_SYSTEM_SQLITE3, db_path],
        input=sql,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise sqlite3.OperationalError(
            f"sqlite3 CLI 执行失败: {result.stderr.strip()}"
        )
    return result.stdout


def execute_query(
    query: str, db_path: str, row_type: Type[sqlite_schema_utils.RowType]
) -> list[sqlite_schema_utils.RowType]:
  """Retrieves all rows from the given SQLite database path.

  Args:
    query: The query to issue.
    db_path: The path to the SQLite database file.
    row_type: The object type that will be created for each retrieved row.

  Returns:
      A list of tuples, each representing an row from the database.
  """
  conn = sqlite3.connect(db_path)
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()
  raw_rows = cursor.execute(query).fetchall()
  conn.close()

  rows = []
  for row in raw_rows:
    row = dict(row)
    rows.append(row_type(**row))  # pytype: disable=bad-return-type
  return rows


def get_rows_from_remote_device(
    table_name: str,
    remote_db_file_path: str,
    row_type: Type[sqlite_schema_utils.RowType],
    env: interface.AsyncEnv,
    timeout_sec: Optional[float] = None,
    n_retries: int = 3,
    app_name: Optional[str] = None,
) -> list[sqlite_schema_utils.RowType]:
  """Retrieves rows from a table in a SQLite database located on a remote Android device.

  This function first copies the database from the remote device to a
  temporary local directory.

  Args:
    table_name: The name of the table from which to retrieve rows.
    remote_db_file_path: The database path on the remote device.
    row_type: The class type corresponding to the table's row structure. Each
      new database needs an equivalent python representation class type.
    env: The Android environment interface used for interacting with the remote
      device.
    timeout_sec: Optional timeout in seconds for the database copy operation.
    n_retries: The number of times to try. This is relevant in cases where a
      database has not been created/being created when an app is launched for
      the first time after clearing the database.
    app_name: Deprecated. Previously used to close the app before pulling.
      Now WAL is merged via device-side PRAGMA checkpoint instead. Kept for
      backward compatibility.

  Returns:
    All rows from the table.

  Raises:
    ValueError: If cannot query table.
  """
  # 在设备端执行 WAL checkpoint，将 WAL 数据合并到主数据库。
  # 注意：am force-stop 并不会触发 SQLite WAL checkpoint，
  # 所以不能依赖杀死应用来合并 WAL。这里直接在设备上执行
  # PRAGMA wal_checkpoint(TRUNCATE)，无需杀死应用。
  _checkpoint_wal_on_device(remote_db_file_path, env, timeout_sec)

  for attempt in range(n_retries):
    try:
      with env.controller.pull_file(
          remote_db_file_path, timeout_sec
      ) as local_db_directory:
        local_db_path = file_utils.convert_to_posix_path(
            local_db_directory, os.path.split(remote_db_file_path)[1]
        )
        return execute_query(
            f"SELECT * FROM {table_name};",
            local_db_path,
            row_type,
        )
    except (sqlite3.OperationalError, FileNotFoundError):
      if attempt < n_retries - 1:
        time.sleep(2.0)
  raise ValueError(
      f"Failed to retrieve rows from {table_name} from"
      f" {remote_db_file_path} after {n_retries} retries. Try increasing the "
      "number of retries."
  )


def table_exists(
    table_name: str,
    remote_db_file_path: str,
    env: interface.AsyncEnv,
) -> bool:
  """Checks if a table exists in a SQLite database on a remote Android device.

  Args:
    table_name: The name of the table from which to retrieve rows.
    remote_db_file_path: The path to the sqlite database on the device.
    env: The environment.

  Returns:
    True if the table exists in the database.
  """
  try:
    get_rows_from_remote_device(
        table_name,
        remote_db_file_path,
        sqlite_schema_utils.GenericRow,
        env,
    )
    return True
  except (FileNotFoundError, ValueError):
    return False


def delete_all_rows_from_table(
    table_name: str,
    remote_db_file_path: str,
    env: interface.AsyncEnv,
    app_name: str,
    timeout_sec: Optional[float] = None,
) -> None:
  """Deletes all rows from a specified table in a SQLite database on a remote Android device.

  Args:
    table_name: Deletes all rows from the table.
    remote_db_file_path: The path to the sqlite database on the device.
    env: The environment.
    app_name: The name of the app that owns the database.
    timeout_sec: Timeout in seconds.
  """
  if not table_exists(table_name, remote_db_file_path, env):
    # If the database was never created, opening the app may create it.
    adb_utils.launch_app(app_name, env.controller)
    time.sleep(7.0)
    adb_utils.close_app(app_name, env.controller)
    # Re-check after launch; some tables (e.g. OsmAnd map_markers, Retro
    # PlaylistEntity) are only created on first user interaction — nothing
    # to delete in that case.
    if not table_exists(table_name, remote_db_file_path, env):
      logger.info(
          "Table %s still does not exist in %s after app launch — skipping delete.",
          table_name, remote_db_file_path,
      )
      return

  # 先杀死应用以获得独占写入权限，再执行设备端 WAL checkpoint
  adb_utils.close_app(app_name, env.controller)
  _checkpoint_wal_on_device(remote_db_file_path, env)

  with env.controller.pull_file(
      remote_db_file_path, timeout_sec
  ) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(remote_db_file_path)[1]
    )

    delete_command = f"DELETE FROM {table_name}"
    if _HAS_FTS:
      conn = sqlite3.connect(local_db_path)
      cursor = conn.cursor()
      cursor.execute(delete_command)
      conn.commit()
      # 强制 WAL checkpoint 并切换到 DELETE journal 模式，
      # 确保 push 回去的单个 .db 文件包含所有数据，
      # 不依赖 WAL/SHM 文件。
      cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
      cursor.execute("PRAGMA journal_mode=DELETE")
      conn.close()
    else:
      _sqlite3_exec(
          local_db_path,
          f"{delete_command};\n"
          "PRAGMA wal_checkpoint(TRUNCATE);\n"
          "PRAGMA journal_mode=DELETE;\n",
      )

    # 删除本地 WAL/SHM 残留文件，确保只 push 干净的主 DB
    for suffix in ("-wal", "-shm"):
      wal_path = local_db_path + suffix
      if os.path.exists(wal_path):
        os.remove(wal_path)

    env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
    adb_utils.close_app(
        app_name, env.controller
    )  # Close app to register the changes.


def insert_rows_to_remote_db(
    rows: list[sqlite_schema_utils.RowType],
    exclude_key: str | None,
    table_name: str,
    remote_db_file_path: str,
    app_name: str,
    env: interface.AsyncEnv,
    timeout_sec: Optional[float] = None,
) -> None:
  """Inserts rows into a SQLite database located on a remote Android device.

  Args:
    rows: The rows to insert into the remote database.
    exclude_key: Name of field to exclude adding to database. Typically an auto
      incrementing key.
    table_name: The name of the table to insert rows into.
    remote_db_file_path: Location of the SQLite database to insert rows into.
    app_name: The name of the app that owns the database.
    env: The environment.
    timeout_sec: Optional timeout in seconds for the database copy operation.
  """
  # 先杀死应用以获得独占写入权限，再执行设备端 WAL checkpoint
  adb_utils.close_app(app_name, env.controller)
  _checkpoint_wal_on_device(remote_db_file_path, env)

  with env.controller.pull_file(
      remote_db_file_path, timeout_sec
  ) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(remote_db_file_path)[1]
    )

    if _HAS_FTS:
      conn = sqlite3.connect(local_db_path)
      cursor = conn.cursor()
      for row in rows:
        insert_command, values = sqlite_schema_utils.insert_into_db(
            row, table_name, exclude_key
        )
        cursor.execute(insert_command, values)
      conn.commit()
      cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
      cursor.execute("PRAGMA journal_mode=DELETE")
      conn.close()
    else:
      sql_parts = ["BEGIN TRANSACTION;"]
      for row in rows:
        insert_command, values = sqlite_schema_utils.insert_into_db(
            row, table_name, exclude_key
        )
        # Replace ? placeholders with formatted literal values
        for val in values:
          insert_command = insert_command.replace("?", _format_sql_value(val), 1)
        sql_parts.append(f"{insert_command};")
      sql_parts.append("COMMIT;")
      sql_parts.append("PRAGMA wal_checkpoint(TRUNCATE);")
      sql_parts.append("PRAGMA journal_mode=DELETE;")
      _sqlite3_exec(local_db_path, "\n".join(sql_parts))

    # 删除本地 WAL/SHM 残留文件
    for suffix in ("-wal", "-shm"):
      wal_path = local_db_path + suffix
      if os.path.exists(wal_path):
        os.remove(wal_path)

    env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
    adb_utils.close_app(app_name, env.controller)
