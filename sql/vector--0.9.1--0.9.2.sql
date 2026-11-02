-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.9.2'" to load this file. \quit

CREATE FUNCTION ann_frozen_cache(OUT entries bigint, OUT bytes bigint, OUT hits bigint, OUT misses bigint,
	OUT evictions bigint) RETURNS record
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

COMMENT ON FUNCTION ann_frozen_cache() IS 'entries, bytes, hits, misses and evictions of the shared cache of predicate results over frozen ann indexes';

CREATE FUNCTION ann_frozen_prewarm(index regclass) RETURNS bigint
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

COMMENT ON FUNCTION ann_frozen_prewarm(regclass) IS 'read a frozen ann index into memory ahead of queries; returns the bytes read';

CREATE FUNCTION ann_frozen_precompute(index regclass, filters text[]) RETURNS bigint
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

COMMENT ON FUNCTION ann_frozen_precompute(regclass, text[]) IS 'evaluate filters over a frozen ann index''s attributes into the shared cache and the GPU worker''s; returns how many were cached';
