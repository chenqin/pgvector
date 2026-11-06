#define _POSIX_C_SOURCE 200809L
#include "cost.h"
#include <assert.h>
#include <unistd.h>

int
main(void)
{
	char		path[] = "/tmp/ann-cost-test-XXXXXX";
	int			fd = mkstemp(path);
	FILE	   *file;
	AnnCostProfile *p;
	char		error[160];
	double		us[ANN_COST_MODES];
	double		expected = 0;

	assert(fd >= 0);
	file = fdopen(fd, "w");
	assert(file);
	fprintf(file, "ANN_COST_PROFILE 5 1000 128 123 8 50 0.2 0.95 4096 256\n");
	for (int a = 0; a < 2; a++)
		for (int b = 0; b < 2; b++)
			for (int c = 0; c < 2; c++)
			{
				double		t = 100 + 100 * a + 10 * b + c;

				fprintf(file, "%d %d %d %.9g %.9g %.9g %.9g %.9g\n", a ? 100 : 10,
						b ? 50 : 5, c ? 16 : 1, t * 2, t, t * 3, t * 4, t * 10);
			}
	fclose(file);
	p = AnnCostLoad(path, error, sizeof(error));
	assert(p && p->setup_us == 50 && p->tuple_us == 0.2 && p->min_recall == 0.95 && p->graph_itopk == 4096 && p->graph_slack == 256);
	assert(AnnCostEstimate(p, 1000, 128, 0x123, sqrt(1000), sqrt(250), 4, us));
	for (unsigned int i = 0; i < p->count; i++)
		expected += log(p->cells[i].us[ANN_COST_SCAN_GPU]) / p->count;
	assert(fabs(us[ANN_COST_SCAN_GPU] - exp(expected)) < 1e-8);
	assert(AnnCostBest(us, 1, 1, 1) == ANN_COST_SCAN_GPU);
	assert(AnnCostBest(us, 1, 1, 0) == ANN_COST_SCAN_HOST);
	assert(!AnnCostEstimate(p, 1000, 128, 0x124, 10, 5, 1, us));
	assert(!AnnCostEstimate(p, 1000, 64, 0x123, 10, 5, 1, us));
	assert(!AnnCostEstimate(p, 1000, 128, 0x123, 101, 5, 1, us));
	assert(!AnnCostEstimate(p, 1000, 128, 0x123, NAN, 5, 1, us));
	p->cells[0].us[ANN_COST_SCAN_GPU] = -1;
	assert(AnnCostEstimate(p, 1000, 128, 0x123, sqrt(1000), sqrt(250), 4, us));
	assert(us[ANN_COST_SCAN_GPU] == -1);
	assert(AnnCostBest(us, 1, 1, 1) == ANN_COST_SCAN_HOST);
	/* Sub-microsecond log latencies are negative, but still qualified. */
	for (unsigned int i = 0; i < p->count; i++)
		p->cells[i].us[ANN_COST_SCAN_GPU] = 0.25;
	assert(AnnCostEstimate(p, 1000, 128, 0x123, sqrt(1000), sqrt(250), 4, us));
	assert(fabs(us[ANN_COST_SCAN_GPU] - 0.25) < 1e-12);
	free(p);
	file = fopen(path, "w");
	assert(file);
	fprintf(file, "ANN_COST_PROFILE 1 1000 128 123 1\n10 5 1 nan 2 3 4 5\n");
	fclose(file);
	assert(AnnCostLoad(path, error, sizeof(error)) == NULL);
	unlink(path);
	puts("cost model: interpolation, availability, qualification, bounds and validation passed");
	return 0;
}
