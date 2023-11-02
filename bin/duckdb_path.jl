import DuckDB
using Pkg.Artifacts

artifacts_toml = dirname(pathof(DuckDB.DuckDB_jll)) * "/../Artifacts.toml"
hash = artifact_hash("DuckDB", artifacts_toml)
duckdb_bin = artifact_path(hash) * "/bin/duckdb"
println(duckdb_bin)
