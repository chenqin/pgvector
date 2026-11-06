-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.9.3'" to load this file. \quit

CREATE TABLE ann_collection (
	name text PRIMARY KEY,
	source regclass NOT NULL,
	attributes text[] NOT NULL,
	placement text,
	delta regclass,
	delta_key name,
	delta_vector name,
	created_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE ann_collection IS 'ann collections: tables split into parts served by GPU nodes, searched together by ann_collection_search';

REVOKE ALL ON ann_collection FROM PUBLIC;
GRANT SELECT ON ann_collection TO PUBLIC;

CREATE TABLE ann_collection_part (
	collection text NOT NULL REFERENCES ann_collection (name) ON DELETE CASCADE,
	part integer NOT NULL,
	index_name text NOT NULL,
	nodes text NOT NULL,
	generation text,
	rows bigint NOT NULL,
	placement_low integer,
	placement_high integer,
	loaded_at timestamptz NOT NULL DEFAULT now(),
	PRIMARY KEY (collection, part)
);

COMMENT ON TABLE ann_collection_part IS 'parts of ann collections: the loaded index, its replica nodes, generation and placement range';

REVOKE ALL ON ann_collection_part FROM PUBLIC;
GRANT SELECT ON ann_collection_part TO PUBLIC;

CREATE TABLE ann_collection_dictionary (
	collection text NOT NULL REFERENCES ann_collection (name) ON DELETE CASCADE,
	attribute text NOT NULL,
	code integer NOT NULL,
	value text NOT NULL,
	PRIMARY KEY (collection, attribute, code),
	UNIQUE (collection, attribute, value)
);

COMMENT ON TABLE ann_collection_dictionary IS 'codes of the strings of ann collections'' text attributes: assigned as parts bring new strings, never reused or reordered';

REVOKE ALL ON ann_collection_dictionary FROM PUBLIC;
GRANT SELECT ON ann_collection_dictionary TO PUBLIC;

CREATE TABLE ann_collection_tombstone (
	collection text NOT NULL REFERENCES ann_collection (name) ON DELETE CASCADE,
	key bigint NOT NULL,
	deleted_at timestamptz NOT NULL DEFAULT now(),
	PRIMARY KEY (collection, key)
);

COMMENT ON TABLE ann_collection_tombstone IS 'keys deleted from ann collections, left out of searches until their partitions are rebuilt without them';

REVOKE ALL ON ann_collection_tombstone FROM PUBLIC;
GRANT SELECT ON ann_collection_tombstone TO PUBLIC;

CREATE TABLE ann_frozen_tombstone (
	source regclass NOT NULL,
	key bigint NOT NULL,
	deleted_at timestamptz NOT NULL DEFAULT now(),
	PRIMARY KEY (source, key)
);

COMMENT ON TABLE ann_frozen_tombstone IS 'keys deleted from frozen tables, left out of the frozen route and GPU searches until the table is rebuilt without them';

REVOKE ALL ON ann_frozen_tombstone FROM PUBLIC;
GRANT SELECT ON ann_frozen_tombstone TO PUBLIC;

CREATE FUNCTION ann_collection_create(name text, source regclass, attributes text[], placement text DEFAULT NULL)
	RETURNS void AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

COMMENT ON FUNCTION ann_collection_create(text, regclass, text[], text) IS 'create an ann collection over a table: the attribute columns its parts were built with, and the attribute they are placed by';

REVOKE ALL ON FUNCTION ann_collection_create(text, regclass, text[], text) FROM PUBLIC;

CREATE FUNCTION ann_collection_load(collection text, part integer, index_name text, path text, nodes text,
	placement_low integer DEFAULT NULL, placement_high integer DEFAULT NULL, unload_previous boolean DEFAULT false)
	RETURNS bigint
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

COMMENT ON FUNCTION ann_collection_load(text, integer, text, text, text, integer, integer, boolean) IS 'load (or with a NULL path attach) an ann collection part on its replica nodes and record it, with unload_previous freeing the index it replaces; returns its rows';

REVOKE ALL ON FUNCTION ann_collection_load(text, integer, text, text, text, integer, integer, boolean) FROM PUBLIC;

CREATE FUNCTION ann_collection_search(collection text, query vector, k integer, filter text DEFAULT NULL,
	OUT key bigint, OUT distance real) RETURNS SETOF record
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE ROWS 10;

COMMENT ON FUNCTION ann_collection_search(text, vector, integer, text) IS 'the k nearest rows of an ann collection matching a filter over its attributes, searched on the parts'' GPU nodes in parallel';

CREATE FUNCTION ann_collection_search_rows(collection text, query vector, k integer, filter text DEFAULT NULL)
	RETURNS SETOF record AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE ROWS 10;

COMMENT ON FUNCTION ann_collection_search_rows(text, vector, integer, text) IS 'as ann_collection_search, with attribute values from the GPU nodes: AS r(key bigint, distance real, <attribute> <type>, ...)';

CREATE FUNCTION ann_collection_delta(collection text, delta regclass, key_column text DEFAULT 'id',
	vector_column text DEFAULT 'embedding') RETURNS void
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

COMMENT ON FUNCTION ann_collection_delta(text, regclass, text, text) IS 'set (or with NULL clear) the table of an ann collection''s rows not yet in a part, searched exactly with each query';

REVOKE ALL ON FUNCTION ann_collection_delta(text, regclass, text, text) FROM PUBLIC;

CREATE FUNCTION ann_index_unload(index_name text, nodes text DEFAULT NULL) RETURNS integer
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

COMMENT ON FUNCTION ann_index_unload(text, text) IS 'unload an index from GPU nodes once its searches finish, as after a part was replaced; returns the nodes that had it';

REVOKE ALL ON FUNCTION ann_index_unload(text, text) FROM PUBLIC;

CREATE FUNCTION ann_collection_delete(collection text, keys bigint[]) RETURNS bigint
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

COMMENT ON FUNCTION ann_collection_delete(text, bigint[]) IS 'delete rows of an ann collection by key: tombstones its parts'' nodes leave out of searches until the partitions are rebuilt; returns how many were new';

REVOKE ALL ON FUNCTION ann_collection_delete(text, bigint[]) FROM PUBLIC;

CREATE FUNCTION ann_frozen_delete(source regclass, keys bigint[]) RETURNS bigint
	AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

COMMENT ON FUNCTION ann_frozen_delete(regclass, bigint[]) IS 'delete rows of a frozen table by key: tombstones the frozen route and its loaded exports leave out until the table is rebuilt; returns how many were new';

REVOKE ALL ON FUNCTION ann_frozen_delete(regclass, bigint[]) FROM PUBLIC;

CREATE FUNCTION ann_frozen_sizes(index regclass, OUT structure text, OUT pages bigint, OUT bytes bigint)
	RETURNS SETOF record AS 'MODULE_PATHNAME' LANGUAGE C STRICT VOLATILE;

COMMENT ON FUNCTION ann_frozen_sizes(regclass) IS 'pages and bytes of each structure of a frozen ann index, with how it keeps the vectors';

CREATE FUNCTION ann_frozen_search_batch(name text, queries vector[], k integer, filter text DEFAULT NULL,
	OUT query_no integer, OUT id bigint, OUT distance real) RETURNS SETOF record
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE ROWS 500;

COMMENT ON FUNCTION ann_frozen_search_batch(text, vector[], integer, text) IS 'search up to 64 vectors with one worker request and a shared predicate and statement snapshot; query_no is the one-based input position';
