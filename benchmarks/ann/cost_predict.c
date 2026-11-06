#include "cost.h"

int
main(int argc, char **argv)
{
	AnnCostProfile *profile;
	char		error[160];
	double		matching,
				k,
				batch;

	if (argc != 2 || (profile = AnnCostLoad(argv[1], error, sizeof(error))) == NULL)
	{
		fprintf(stderr, "usage: cost-predict PROFILE (matching k batch on stdin)\n");
		return 1;
	}
	while (scanf("%lf %lf %lf", &matching, &k, &batch) == 3)
	{
		double		us[ANN_COST_MODES] = {-1, -1, -1, -1, -1};
		int			valid = AnnCostEstimate(profile, profile->rows, profile->dimensions, profile->generation,
											matching, k, batch, us);
		int			best = valid ? AnnCostBest(us, 1, 1, 1) : -1;

		printf("%d %.9g %.9g %.9g %.9g %.9g\n", best, us[0], us[1], us[2], us[3], us[4]);
	}
	free(profile);
	return 0;
}
