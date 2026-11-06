#include "udf.h"

#include <stdlib.h>
#include <string.h>

#include "filter.h"

static uint64_t
Read(const unsigned char *data, int bytes)
{
	uint64_t	value = 0;

	for (int i = bytes - 1; i >= 0; i--)
		value = (value << 8) | data[i];
	return value;
}

static int
ValueCompare(const void *left, const void *right)
{
	int32_t		a = *(const int32_t *) left;
	int32_t		b = *(const int32_t *) right;

	return (a > b) - (a < b);
}

int
AnnUdfCompile(const unsigned char *program, size_t size,
			  int (*column) (void *arg, const char *name), void *arg,
			  AnnUdfStep *steps, uint32_t maxSteps, uint32_t *nsteps,
			  int32_t *values, uint32_t maxValues, uint32_t *nvalues)
{
	size_t		offset = 0;
	uint32_t	stepCount = *nsteps;
	uint32_t	valueCount = *nvalues;

	if (!AnnFilterCheck(program, size))
		return ANN_FILTER_INVALID;
	while (offset < size)
	{
		int			op = program[offset];
		AnnUdfStep	step;

		memset(&step, 0, sizeof(step));
		step.op = (uint8_t) op;
		if (stepCount >= maxSteps)
			return ANN_UDF_TOO_LARGE;
		switch (op)
		{
			case ANN_FILTER_AND:
			case ANN_FILTER_OR:
			case ANN_FILTER_NOT:
			case ANN_FILTER_ALL:
			case ANN_FILTER_NONE:
				offset++;
				break;
			case ANN_FILTER_BITMAP:
				return ANN_UDF_UNSUPPORTED;
			default:
				{
					uint32_t	nameLength = program[offset + 1];
					char		name[64];
					int			c;
					const unsigned char *at = program + offset + 3 + nameLength;

					memcpy(name, program + offset + 2, nameLength);
					name[nameLength] = '\0';
					if ((c = column(arg, name)) < 0 || c >= ANN_UDF_MAX_COLUMNS)
						return ANN_FILTER_UNKNOWN_COLUMN;
					step.column = (uint8_t) c;
					step.negate = program[offset + 2 + nameLength];
					offset += 3 + nameLength;
					if (op == ANN_FILTER_RANGE)
					{
						step.low = (int32_t) (uint32_t) Read(at, 4);
						step.high = (int32_t) (uint32_t) Read(at + 4, 4);
						offset += 8;
					}
					else if (op == ANN_FILTER_VALUES)
					{
						uint32_t	n = (uint32_t) Read(at, 4);
						uint32_t	kept = 0;

						if ((uint64_t) valueCount + n > maxValues)
							return ANN_UDF_TOO_LARGE;
						for (uint32_t i = 0; i < n; i++)
							values[valueCount + i] = (int32_t) (uint32_t) Read(at + 4 + (size_t) i * 4, 4);
						qsort(values + valueCount, n, sizeof(int32_t), ValueCompare);
						/* Unique, without NULL: VALUES never matches it. */
						for (uint32_t i = 0; i < n; i++)
							if (values[valueCount + i] != ANN_FILTER_NULL_VALUE &&
								(kept == 0 || values[valueCount + i] != values[valueCount + kept - 1]))
								values[valueCount + kept++] = values[valueCount + i];
						step.first = valueCount;
						step.count = kept;
						valueCount += kept;
						offset += 4 + (size_t) n * 4;
					}
				}
				break;
		}
		steps[stepCount++] = step;
	}
	*nsteps = stepCount;
	*nvalues = valueCount;
	return ANN_FILTER_OK;
}

static int
LeafMatch(const AnnUdfStep *step, const int32_t *values, int32_t value)
{
	int			match;

	if (step->op == ANN_FILTER_NULL)
		match = value == ANN_FILTER_NULL_VALUE;
	else if (value == ANN_FILTER_NULL_VALUE)
		return 0;				/* negated or not, comparisons skip NULL */
	else if (step->op == ANN_FILTER_RANGE)
		match = step->low <= value && value <= step->high;
	else
	{
		uint32_t	lo = step->first,
					hi = step->first + step->count;

		while (lo < hi)
		{
			uint32_t	mid = lo + (hi - lo) / 2;

			if (values[mid] < value)
				lo = mid + 1;
			else
				hi = mid;
		}
		match = lo < step->first + step->count && values[lo] == value;
	}
	return step->negate ? !match : match;
}

int
AnnUdfMatch(const AnnUdfStep *steps, uint32_t nsteps, const int32_t *values,
			const int32_t *const *columns, uint64_t row)
{
	uint64_t	stack = 0;
	int			depth = 0;

	for (uint32_t i = 0; i < nsteps; i++)
	{
		const AnnUdfStep *step = &steps[i];
		uint64_t	bit;

		switch (step->op)
		{
			case ANN_FILTER_AND:
			case ANN_FILTER_OR:
				{
					uint64_t	b = (stack >> (depth - 1)) & 1;
					uint64_t	a = (stack >> (depth - 2)) & 1;

					depth -= 2;
					bit = step->op == ANN_FILTER_AND ? (a & b) : (a | b);
				}
				break;
			case ANN_FILTER_NOT:
				bit = ((stack >> (depth - 1)) & 1) ^ 1;
				depth--;
				break;
			case ANN_FILTER_ALL:
				bit = 1;
				break;
			case ANN_FILTER_NONE:
				bit = 0;
				break;
			default:
				bit = (uint64_t) LeafMatch(step, values, columns[step->column][row]);
				break;
		}
		stack = (stack & ~(UINT64_C(1) << depth)) | (bit << depth);
		depth++;
	}
	return depth == 1 && (stack & 1);
}

double
AnnUdfEstimate(const AnnUdfStep *steps, uint32_t nsteps, const int32_t *values,
			   uint64_t (*rows) (void *arg, int column, int32_t low, int32_t high),
			   void *arg, uint64_t total)
{
	double		stack[ANN_FILTER_MAX_DEPTH];
	int			depth = 0;

	if (total == 0)
		return 0;
	for (uint32_t i = 0; i < nsteps; i++)
	{
		const AnnUdfStep *step = &steps[i];
		double		p;

		switch (step->op)
		{
			case ANN_FILTER_AND:
			case ANN_FILTER_OR:
				{
					double		b = stack[--depth];
					double		a = stack[--depth];

					p = step->op == ANN_FILTER_AND ? a * b : a + b - a * b;
				}
				break;
			case ANN_FILTER_NOT:
				p = 1.0 - stack[--depth];
				break;
			case ANN_FILTER_ALL:
				p = 1.0;
				break;
			case ANN_FILTER_NONE:
				p = 0.0;
				break;
			default:
				{
					double		nulls = (double) rows(arg, step->column, ANN_FILTER_NULL_VALUE,
													  ANN_FILTER_NULL_VALUE);
					double		match = 0;

					if (step->op == ANN_FILTER_NULL)
						match = nulls;
					else if (step->op == ANN_FILTER_RANGE)
					{
						int32_t		low = step->low > ANN_FILTER_NULL_VALUE ? step->low : ANN_FILTER_NULL_VALUE + 1;

						if (low <= step->high)
							match = (double) rows(arg, step->column, low, step->high);
					}
					else
						for (uint32_t v = 0; v < step->count; v++)
							match += (double) rows(arg, step->column, values[step->first + v],
												   values[step->first + v]);
					if (step->negate)
						match = (double) total - match - (step->op == ANN_FILTER_NULL ? 0 : nulls);
					p = match / (double) total;
				}
				break;
		}
		stack[depth++] = p < 0 ? 0 : p > 1 ? 1 : p;
	}
	return depth == 1 ? stack[0] * (double) total : 0;
}
