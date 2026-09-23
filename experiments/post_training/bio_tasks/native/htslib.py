# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct HTSlib faidx access, independent of the SAMtools executable."""

import shlex
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import table

PROGRAM = r"""
#include <stdio.h>
#include <stdlib.h>
#include <htslib/faidx.h>
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    faidx_t *index = fai_load(argv[1]);
    if (!index) return 3;
    int length;
    char *sequence = faidx_fetch_seq(index, argv[2], atoi(argv[3])-1, atoi(argv[4])-1, &length);
    if (!sequence || length < 0) return 4;
    puts(sequence);
    free(sequence);
    fai_destroy(index);
    return 0;
}
"""


def solve_htslib(inputs: Path, work: Path) -> list[dict]:
    source = work / "fetch.c"
    source.write_text(PROGRAM)
    flags = execute(["pkg-config", "--cflags", "--libs", "htslib"], work, "flags.txt")
    executable = work / "fetch"
    execute(["cc", str(source), "-o", str(executable), *shlex.split(flags.read_text())], work, "compile.log")
    answer = []
    for row in table(inputs / "regions.csv"):
        output = execute(
            [str(executable), str(inputs / "reference.fa"), row["contig"], row["start"], row["end"]],
            work,
            row["region"] + ".txt",
        )
        answer.append({"id": row["region"], "sequence": output.read_text().strip()})
    return answer
