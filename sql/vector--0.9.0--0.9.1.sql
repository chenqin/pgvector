-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.9.1'" to load this file. \quit

-- frozen indexes (some 0.9.0 installs already have these functions)

CREATE OR REPLACE FUNCTION ann_frozen_guard() RETURNS trigger
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE OR REPLACE FUNCTION ann_freeze(rel regclass) RETURNS boolean
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

CREATE OR REPLACE FUNCTION ann_frozen_info(index regclass, OUT generation text, OUT rows bigint, OUT dimensions integer,
	OUT key_column text, OUT attributes text[], OUT distinct_values bigint[], OUT pages bigint) RETURNS record
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT STABLE;

CREATE OR REPLACE FUNCTION ann_frozen_keys(index regclass, attribute text, attribute_values integer[]) RETURNS SETOF bigint
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT STABLE ROWS 1000;

CREATE OR REPLACE FUNCTION ann_index_export(index regclass, path text) RETURNS bigint
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

REVOKE ALL ON FUNCTION ann_index_export(regclass, text) FROM PUBLIC;

ALTER TABLE ann_index_catalog ADD COLUMN IF NOT EXISTS generation text;

COMMENT ON COLUMN ann_index_catalog.generation IS 'generation of the frozen ann index the loaded files were exported from';

CREATE FUNCTION ann_frozen_filter(index regclass, filter text, OUT eligible bigint, OUT bitmap bytea) RETURNS record
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT STABLE;

COMMENT ON FUNCTION ann_frozen_filter(regclass, text) IS 'rows of a filter over a frozen ann index''s INCLUDE attributes, from its postings and bitset pages';

CREATE FUNCTION ann_index_filter(name text, filter text, OUT eligible bigint, OUT bitmap bytea) RETURNS record
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

COMMENT ON FUNCTION ann_index_filter(text, text) IS 'rows of a filter over an exported frozen index''s attributes, evaluated on the GPU node';
