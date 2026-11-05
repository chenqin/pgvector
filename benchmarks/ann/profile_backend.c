/*
 * Session-preload helper for debugger sampling of user-owned test backends
 * on Linux with Yama ptrace restrictions. It changes only the calling
 * process, not the host's perf/ptrace settings. No SQL functions are added.
 */
#include "postgres.h"
#include "fmgr.h"
#include <sys/prctl.h>

PG_MODULE_MAGIC;
PGDLLEXPORT void _PG_init(void);

void
_PG_init(void)
{
	if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) != 0)
		elog(ERROR, "could not enable debugger sampling of this benchmark backend: %m");
}
