#!/bin/bash

./bin/duckdb <<SQL
PRAGMA enable_progress_bar;
INSTALL 'json';
LOAD 'json';

CREATE TEMP TABLE results AS SELECT * FROM read_ndjson_auto('$1');

COPY (
    SELECT *
    FROM results 
    WHERE quantity = 'runtime'
) TO '$2' (FORMAT 'parquet');
SQL
