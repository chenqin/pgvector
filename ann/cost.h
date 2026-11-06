#ifndef ANN_COST_H
#define ANN_COST_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANN_COST_MAX_CELLS 1024
#define ANN_COST_MODES 5
#define ANN_COST_SCAN_HOST 0
#define ANN_COST_SCAN_GPU 1
#define ANN_COST_GRAPH_HOST 2
#define ANN_COST_GRAPH_GPU 3
#define ANN_COST_LOCAL 4

typedef struct AnnCostCell
{
	double		matching;
	double		k;
	double		batch;
	double		us[ANN_COST_MODES];
} AnnCostCell;

typedef struct AnnCostProfile
{
	uint64_t	rows;
	uint64_t	generation;
	unsigned int dimensions;
	unsigned int count;
	double		setup_us;
	double		tuple_us;
	double		min_recall;
	unsigned int graph_itopk;
	unsigned int graph_slack;
	AnnCostCell cells[ANN_COST_MAX_CELLS];
} AnnCostProfile;

/* A profile is explicit deployment configuration, never inferred from timing noise. */
static inline AnnCostProfile *
AnnCostLoad(const char *path, char *error, size_t capacity)
{
	FILE	   *file = fopen(path, "r");
	AnnCostProfile *profile = NULL;
	char		line[512];
	char		extra;
	unsigned int version = 0;
	int			fields;
	unsigned long long rows = 0,
				generation = 0;

	if (file == NULL)
	{
		snprintf(error, capacity, "could not open cost profile");
		return NULL;
	}
	profile = calloc(1, sizeof(*profile));
	if (profile == NULL)
		goto invalid;
	if (fgets(line, sizeof(line), file) == NULL)
		goto invalid;
	profile->min_recall = 0.95;
	fields = sscanf(line, "ANN_COST_PROFILE %u %llu %u %llx %u %lf %lf %lf %u %u %c", &version, &rows,
					&profile->dimensions, &generation, &profile->count,
					&profile->setup_us, &profile->tuple_us, &profile->min_recall, &profile->graph_itopk, &profile->graph_slack, &extra);
	if (!((version == 1 && fields == 5) || (version == 2 && fields == 7) || (version == 3 && fields == 8) || (version == 4 && fields == 9) || (version == 5 && fields == 10)) ||
		rows == 0 || rows >= UINT32_MAX || generation == 0 || profile->dimensions == 0 ||
		profile->count == 0 || profile->count > ANN_COST_MAX_CELLS ||
		!isfinite(profile->setup_us) || profile->setup_us < 0 ||
		!isfinite(profile->tuple_us) || profile->tuple_us < 0 ||
		!isfinite(profile->min_recall) || profile->min_recall <= 0 || profile->min_recall > 1 ||
		(version >= 4 && (profile->graph_itopk < 32 || profile->graph_itopk > 65536 || profile->graph_itopk % 32)) || profile->graph_slack > 10000)
		goto invalid;
	profile->rows = (uint64_t) rows;
	profile->generation = (uint64_t) generation;
	for (unsigned int i = 0; i < profile->count; i++)
	{
		AnnCostCell *cell = &profile->cells[i];

		if (fgets(line, sizeof(line), file) == NULL ||
			sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %c", &cell->matching, &cell->k,
				   &cell->batch, &cell->us[0], &cell->us[1], &cell->us[2], &cell->us[3], &cell->us[4], &extra) != 8 ||
			!isfinite(cell->matching) || cell->matching < 1 || cell->matching > (double) rows ||
			!isfinite(cell->k) || cell->k < 1 || cell->k > 10000 ||
			!isfinite(cell->batch) || cell->batch < 1 || cell->batch > 64)
			goto invalid;
		for (unsigned int m = 0; m < ANN_COST_MODES; m++)
			if (!isfinite(cell->us[m]) || ((cell->us[m] < 1e-6 || cell->us[m] > 1e12) && cell->us[m] != -1))
				goto invalid;
		for (unsigned int j = 0; j < i; j++)
			if (profile->cells[j].matching == cell->matching &&
				profile->cells[j].k == cell->k && profile->cells[j].batch == cell->batch)
				goto invalid;
	}
	if (fgets(line, sizeof(line), file) != NULL)
		goto invalid;
	fclose(file);
	return profile;
invalid:
	snprintf(error, capacity, "invalid or oversized ANN cost profile");
	fclose(file);
	free(profile);
	return NULL;
}

static inline int
AnnCostMatches(const AnnCostProfile *profile, uint64_t rows, uint32_t dimensions, uint64_t generation)
{
	return profile && profile->rows == rows && profile->dimensions == dimensions && profile->generation == generation;
}

/*
 * Interpolate log latency in log matching-rows/k/batch coordinates. Do not
 * extrapolate or interpolate through an unqualified (negative) route sample.
 * Times are per request, not per vector; LOCAL is a single-vector SQL time.
 */
static inline int
AnnCostEstimate(const AnnCostProfile *profile, uint64_t rows, uint32_t dimensions, uint64_t generation,
				double matching, double k, double batch, double us[ANN_COST_MODES])
{
	double		values[3] = {matching, k, batch};
	double		lower[3] = {0, 0, 0};
	double		upper[3] = {INFINITY, INFINITY, INFINITY};
	double		weight[3];
	int			qualified[ANN_COST_MODES] = {1, 1, 1, 1, 1};

	if (!AnnCostMatches(profile, rows, dimensions, generation))
		return 0;
	for (unsigned int i = 0; i < profile->count; i++)
	{
		double		point[3] = {profile->cells[i].matching, profile->cells[i].k, profile->cells[i].batch};

		for (int d = 0; d < 3; d++)
		{
			if (point[d] <= values[d] && point[d] > lower[d])
				lower[d] = point[d];
			if (point[d] >= values[d] && point[d] < upper[d])
				upper[d] = point[d];
		}
	}
	for (int d = 0; d < 3; d++)
	{
		if (!isfinite(values[d]) || lower[d] == 0 || !isfinite(upper[d]))
			return 0;
		weight[d] = lower[d] == upper[d] ? 0 : log(values[d] / lower[d]) / log(upper[d] / lower[d]);
	}
	for (int m = 0; m < ANN_COST_MODES; m++)
		us[m] = 0;
	for (unsigned int corner = 0; corner < 8; corner++)
	{
		double		point[3];
		double		w = 1;
		const AnnCostCell *cell = NULL;

		for (int d = 0; d < 3; d++)
		{
			int			hi = (corner >> d) & 1;

			point[d] = hi ? upper[d] : lower[d];
			w *= hi ? weight[d] : 1 - weight[d];
		}
		if (w == 0)
			continue;
		for (unsigned int i = 0; i < profile->count; i++)
			if (profile->cells[i].matching == point[0] && profile->cells[i].k == point[1] &&
				profile->cells[i].batch == point[2])
			{
				cell = &profile->cells[i];
				break;
			}
		if (cell == NULL)
			return 0;
		for (int m = 0; m < ANN_COST_MODES; m++)
			if (cell->us[m] < 0)
				qualified[m] = 0;
			else
				us[m] += w * log(cell->us[m]);
	}
	for (int m = 0; m < ANN_COST_MODES; m++)
		us[m] = qualified[m] ? exp(us[m]) : -1;
	return 1;
}

static inline int
AnnCostBest(const double us[ANN_COST_MODES], int scan, int graph, int gpu)
{
	int			best = -1;

	for (int m = 0; m < ANN_COST_LOCAL; m++)
		if ((m < ANN_COST_GRAPH_HOST ? scan : graph) && (gpu || !(m & 1)) &&
			us[m] > 0 && (best < 0 || us[m] < us[best]))
			best = m;
	return best;
}

static inline const char *
AnnCostName(int mode)
{
	static const char *names[] = {"NVFP4 scan, CPU rerank", "NVFP4 scan, GPU rerank",
	"CAGRA, CPU rerank", "CAGRA, GPU rerank", "local SQL"};

	return mode >= 0 && mode < ANN_COST_MODES ? names[mode] : "legacy";
}

static inline uint64_t
AnnCostGatherLimit(const AnnCostProfile *profile)
{
	uint64_t	limit = 0;

	if (profile)
		for (unsigned int i = 0; i < profile->count; i++)
		{
			const AnnCostCell *c = &profile->cells[i];
			int			best = AnnCostBest(c->us, 1, 1, 1);

			if (best >= 0 && best < ANN_COST_GRAPH_HOST && c->matching > (double) limit)
				limit = (uint64_t) c->matching;
		}
	return limit;
}
#endif
