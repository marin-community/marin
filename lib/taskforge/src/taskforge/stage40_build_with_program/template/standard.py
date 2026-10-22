# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The standard build program: research, fixtures, machine, grader, instructions, assembly.

The author shows this module to the model as the program to adapt, so it keeps the rules an
authored program keeps. Its module level holds imports, steps and ``PROGRAM`` only. Each step
holds the prompts it sends in its body, where its memo key covers them.

The steps, in run order:

1. ``sources`` calls ``research`` for each research item of the proposal, concurrently. The notes
   feed the later steps.
2. ``fixtures`` writes the solver-visible files and the private reference data for the solver
   machine the proposal declares (``solver_machine``), and lists the PyPI packages of a package
   machine.
3. ``requirements`` returns the declared task machine: none (``draft.NO_MACHINE``, the solver answers
   in the conversation), ShellSim, Taskforge's grader-base image for CPython, or a package machine
   (``draft.requirements(packages=...)``), which needs no image. A container proposal gets a
   ``DockerBuild``, which ``publish_image`` pushes; the step's output memoizes the pinned reference.
   ``run_build`` refuses a task whose machine is not the declared one.
4. ``grader`` writes the grader of the kind the proposal declares: a decision-model grader
   (``draft.decision_grader``) by default, which ``calibrate_decision_grader`` reads on its reference and
   three equivalent variants and either sends back (a known-correct candidate below "partially
   matches") or calibrates; an LLM judge (``draft.judge_grader``) for a reply whose prose is the
   product; an in-process verifyit grader (exact, numeric, math, mcq) for one value in an
   unambiguous format; otherwise a Python grader program in a verifier machine. A judge calls a model,
   which no build has, so it is not prototyped. For the other kinds, before it returns,
   the step grades its reference answer and the empty answer with ``grade_candidate``. A reference below
   ``FULL_CREDIT``, a reward outside the grader's score range, or an empty answer above the grader's
   declared ``rejection_ceiling`` goes back to the model, and after ``ATTEMPTS`` requests the build
   fails. A reference whose grader timed out raises ``BuildExpired`` at once: ``grade_candidate`` has
   already graded it a second time on fresh machines, so the grader command was slow twice. Whether
   the grader or Taskforge was slow is unknown until someone reads the record, so the workpiece is held
   rather than sent back for a rewritten grader. The returned ``Grader`` holds the reference answer,
   the ceiling and every graded candidate.
5. ``instructions`` writes the solver-facing prompt around the grader's answer contract and sends
   back one that leaks a graded answer, runs past ``INSTRUCTION_WORD_LIMIT`` words, copies file
   lines, or asks for work by hand.
6. ``assemble``, the sink decorated with ``@rl_env_build_sink``, returns the ``TaskSpec``.

Validation lowers the task for its host and writes the controls from the ``Grader``.
"""

import asyncio

from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec

from taskforge.sandbox.images import GRADER_BASE_IMAGE, DockerBuild
from taskforge.spec import draft
from taskforge.spec.decision import DecisionGrader
from taskforge.spec.verdict import GRADER_VERDICT_PATH, grader_checks, reward_in_range
from taskforge.stage10_proposal.model import SolverMachine, TaskProposal
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build_with_program.context import BuildExpired, BuildFailure, ExpiredClock, RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import Fixtures, GradedCandidate, Grader, Instructions, Sources
from taskforge.stage40_build_with_program.sandbox_effects import (
    calibrate_decision_grader,
    grade_candidate,
    publish_image,
)
from taskforge.stage40_build_with_program.sdk import (
    ANSWER_FORMAT,
    DOCKER_IMAGE_REQUIREMENTS,
    FULL_CREDIT,
    INSTRUCTION_WORD_LIMIT,
    NUMERIC_LITERALS,
    SOURCE_DATASET,
    SUBMISSION,
    WORKDIR,
    WORKSPACE,
    DockerfileDraft,
    FixturesDraft,
    GraderDraft,
    InstructionsDraft,
    accept,
    environment_facts,
    files_text,
    grade_diagnostics,
    graded_candidate,
    grader_contract,
    grader_package,
    instruction_problem,
    machine_facts,
    output_paths,
    packages_declared,
    research,
    structured_until,
    task_file,
    task_messages,
)
from taskforge.stage40_build_with_program.step_store import SDK_VERSION


@rl_env_build_step
async def sources(context: RLEnvBuildContext, proposal: TaskProposal) -> Sources:
    findings = await asyncio.gather(*(research(context, proposal, i) for i in range(len(proposal.header.research))))
    notes = "\n\n".join(f"### research[{f.index}]: {f.purpose}\n{f.notes}" for f in findings)
    return Sources(notes=notes, findings=tuple(findings))


@rl_env_build_step
async def fixtures(context: RLEnvBuildContext, proposal: TaskProposal, sources: Sources) -> Fixtures:
    request = (
        f"{environment_facts(proposal.header)}\n\n"
        f"# Research notes\n\n{sources.notes or '(none)'}\n\n"
        "Write the task fixtures from the proposal's Build plan. The task's data lives in solver-visible "
        f"files under {WORKSPACE}/, never in the instruction: tables, datasets, logs and documents are files "
        "(a reasoning task whose inputs are a few short values may ship no files). Grader-only reference data "
        "goes in private_files. Compute every value exactly; the facts field must state every value the grader "
        "will check."
    )
    submitted = await structured_until(
        context, task_messages(proposal, request), FixturesDraft, packages_declared(proposal)
    )
    return Fixtures(
        agent_files=tuple(task_file(f) for f in submitted.agent_files),
        private_files=tuple(task_file(f) for f in submitted.private_files),
        facts=submitted.facts,
        packages=tuple(submitted.packages),
    )


@rl_env_build_step
async def requirements(
    context: RLEnvBuildContext, proposal: TaskProposal, fixtures: Fixtures
) -> EnvironmentRequirements:
    machine = proposal.header.solver_machine
    if machine is SolverMachine.NONE:
        return draft.NO_MACHINE
    if machine is not None:
        image = GRADER_BASE_IMAGE if machine is SolverMachine.CPYTHON else None
        return draft.requirements(image=image, packages=fixtures.packages, workdir=WORKDIR)
    request = (
        f"Write the task image: a Dockerfile (WORKDIR {WORKDIR}, no network at run time) and any build-context "
        f"files it copies. {DOCKER_IMAGE_REQUIREMENTS} The image also provides python3: the grader runs in a "
        "fresh container from it. The solver-visible files below are installed by the task, not the image.\n\n"
        f"{files_text(fixtures.agent_files, '/')}"
    )
    submitted = await structured_until(context, task_messages(proposal, request), DockerfileDraft, accept)
    files = (draft.file("Dockerfile", submitted.dockerfile), *(task_file(f) for f in submitted.context_files))
    try:
        build = DockerBuild(files=files)
    except ValueError as error:
        raise BuildFailure(f"image context: {error}") from error
    return draft.requirements(image=await publish_image(context, build), workdir=WORKDIR)


@rl_env_build_step
async def grader(
    context: RLEnvBuildContext, proposal: TaskProposal, fixtures: Fixtures, requirements: EnvironmentRequirements
) -> Grader:
    request = (
        f"{machine_facts(requirements)}\n\n{grader_contract(requirements)}\n\n"
        f"# Fixture facts\n\n{fixtures.facts}\n\n"
        f"# Solver-visible files\n\n{files_text(fixtures.agent_files, '/')}\n\n"
        f"# Private data files already shipped with a grader script\n\n{files_text(fixtures.private_files, '')}\n\n"
        f"{SUBMISSION}\n\n{NUMERIC_LITERALS}\n\n"
        "Write the grader from the proposal's 'Grader design and controls' section, of the kind its verification "
        f"names: {proposal.header.verification}. decision_model is kind=decision_model: the proposal's rubric, each "
        "item stated as its meaning with the reference values, tolerances, units and accepted equivalences, a "
        "reference reply that delivers what the deliverable statement asks for in the form a real requester "
        "would accept, and three equivalent variants of it; the answer contract is that deliverable statement, "
        "with no answer block. llm_as_judge is kind=llm_as_judge: the same rubric and a reference reply, for a "
        "reply whose prose is the product. answer_in_box is an in-process kind (exact, "
        "numeric, math or mcq) for one value in an unambiguous format. code is kind=script, for a deliverable with "
        "a strict structure the requester specified whose graded parts are all deterministic, or for what only "
        "execution can judge; never for prose. Rewards lie in [0, 1]. Give partial credit only where the "
        "proposal does, and declare rejection_ceiling as the most a submission wrong in substance can collect "
        "under that scheme (0 without partial credit). The reference reply and files must earn full credit."
    )
    prototyped: list[GradedCandidate] = []
    calibrated: list[DecisionGrader] = []

    async def decision_problem(submitted: GraderDraft) -> str | None:
        uncalibrated = grader_package(submitted, fixtures, requirements)
        assert isinstance(uncalibrated, DecisionGrader)
        calibration = await calibrate_decision_grader(context, uncalibrated, submitted.equivalent_variants)
        prototyped.extend(calibration.candidates)
        if calibration.problem is not None:
            return (
                f"{calibration.problem}. Either the rubric rejects a correct answer (state the equivalent forms it "
                "must accept) or a variant is not equivalent to the reference (replace it)"
            )
        calibrated.append(calibration.grader)
        return None

    async def problem(submitted: GraderDraft) -> str | None:
        if submitted.kind == "decision_model":
            return await decision_problem(submitted)
        if submitted.kind == "llm_as_judge":
            # A judge calls a model and a build has none: validation's controls are its first grades.
            try:
                grader_package(submitted, fixtures, requirements)
            except ValueError as error:
                return str(error)
            return None
        if submitted.kind == "script" and not submitted.script.strip():
            return "kind=script needs the script source"
        reference_files = tuple(task_file(f) for f in submitted.reference_files)
        try:
            package = grader_package(submitted, fixtures, requirements)
            assert isinstance(package, GraderPackage)
            paths = output_paths(package, reference_files)
            reference = await grade_candidate(
                context,
                requirements,
                package,
                AnswerType.TEXT,
                ANSWER_FORMAT,
                "(instruction)",
                submitted.reference_reply,
                files=fixtures.agent_files,
                workspace=reference_files,
                output_paths=paths,
            )
        except (ValueError, BuildFailure) as error:
            return str(error)
        if reference.status == Outcome.INFRA_ERROR and reference.failure is GradingFailure.TIMEOUT:
            raise BuildExpired(
                "the grader command timed out on the reference answer twice, each time on fresh machines; the workpiece "
                f"is held for inspection: {grade_diagnostics(reference)}",
                ExpiredClock.GRADER,
            )
        prototyped.append(graded_candidate(submitted.reference_reply, reference_files, reference))
        if reference.status == Outcome.GRADED and not reward_in_range(reference):
            return (
                f"the reference answer got reward {reference.reward}, outside the grader's score range "
                f"[{reference.score_min}, {reference.score_max}]; a grader returns a reward in that range"
            )
        if reference.status != Outcome.GRADED or (reference.reward or 0.0) < FULL_CREDIT:
            return (
                f"the reference answer was graded {reference.status} reward={reference.reward}: "
                f"{grade_diagnostics(reference)}"
            )
        if submitted.kind == "script":
            try:
                checks = grader_checks(reference)
            except ValueError as error:
                return f"the grader's verdict has malformed checks: {error}"
            if not checks:
                return f"the grader wrote no per-check results: put a non-empty detail.checks in {GRADER_VERDICT_PATH}"
        empty = await grade_candidate(
            context,
            requirements,
            package,
            AnswerType.TEXT,
            ANSWER_FORMAT,
            "(instruction)",
            "",
            files=fixtures.agent_files,
            output_paths=paths,
        )
        prototyped.append(graded_candidate("", (), empty))
        if empty.status == Outcome.GRADED and (empty.reward or 0.0) > submitted.rejection_ceiling:
            return (
                f"an empty answer got reward {empty.reward}; it must get at most the declared rejection_ceiling "
                f"{submitted.rejection_ceiling}"
            )
        return None

    submitted = await structured_until(context, task_messages(proposal, request), GraderDraft, problem)
    decides = submitted.kind == "decision_model"
    return Grader(
        package=calibrated[-1] if decides else grader_package(submitted, fixtures, requirements),
        answer_contract=submitted.answer_contract,
        reference_reply=submitted.reference_reply,
        reference_files=tuple(task_file(f) for f in submitted.reference_files),
        secret_values=tuple(value for value in submitted.secret_values if value.strip()),
        rejection_ceiling=submitted.rejection_ceiling,
        prototyped=tuple(prototyped),
    )


@rl_env_build_step
async def instructions(
    context: RLEnvBuildContext,
    proposal: TaskProposal,
    fixtures: Fixtures,
    requirements: EnvironmentRequirements,
    grader: Grader,
) -> Instructions:
    request = (
        f"{machine_facts(requirements)}\n\n"
        f"# Solver-visible files\n\n{files_text(fixtures.agent_files, '/')}\n\n"
        f"# Answer format the solver must follow\n\n{grader.answer_contract}\n\n{SUBMISSION}\n\n"
        "Write the solver-facing instruction from the proposal's Task section: the scenario, every input the "
        "solver needs that is not in a file, the deliverable, and the answer contract verbatim. Rules:\n"
        "- The data lives in the files: name each file the solver needs and never copy its contents, tables or "
        "datasets into the instruction.\n"
        f"- At most {INSTRUCTION_WORD_LIMIT} words, not counting the verbatim answer contract.\n"
        "- Never ask for a deliverable the grader does not read, and never ask the solver to compute by hand.\n"
        "- Describe the deliverable and its format, never the grading: no grader, checker, parser, rubric, "
        "tests, scoring, credit, tolerances, or what is graded, read or ignored. Restate any such line of the "
        "answer contract as a format requirement. Never include a graded answer or hints that give it away.\n"
        "- The answer contract names every literal the answer format requires (headings, terms, field names, "
        "units), so a solver can meet every check from the instruction alone.\n"
        "- Do not describe the shell tool or the submission instruction: RolloutEngine adds both."
    )

    async def problem(submitted: InstructionsDraft) -> str | None:
        return instruction_problem(f"{submitted.system}\n{submitted.instruction}", fixtures, grader)

    submitted = await structured_until(context, task_messages(proposal, request), InstructionsDraft, problem)
    return Instructions(system=submitted.system.strip() or None, instruction=submitted.instruction)


@rl_env_build_sink
async def assemble(
    context: RLEnvBuildContext,
    proposal: TaskProposal,
    fixtures: Fixtures,
    requirements: EnvironmentRequirements,
    grader: Grader,
    instructions: Instructions,
) -> TaskSpec:
    header = proposal.header
    try:
        return draft.assemble(
            task_id=context.workpiece_id,
            instruction=instructions.instruction,
            answer_type=AnswerType.TEXT,
            answer_format=ANSWER_FORMAT,
            grader=grader.package,
            source=Source(
                dataset=SOURCE_DATASET, revision=proposal.digest, row=header.id, importer_revision=SDK_VERSION
            ),
            environment=requirements,
            files=fixtures.agent_files,
            output_paths=output_paths(grader.package, grader.reference_files),
            system=instructions.system,
            tags=(header.environment, header.verification),
        )
    except ValueError as error:
        raise BuildFailure(str(error)) from error


PROGRAM = rl_env_build_program(sources, fixtures, requirements, grader, instructions, sink=assemble)
