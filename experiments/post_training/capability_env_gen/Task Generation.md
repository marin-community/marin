We want to generate realistic RL environments for all of the “capabilities” listed in catalog.json. To do this you will need to use GLM 5.3 agents (we have inference capacity to run hundreds at once in an agentic mode and thousands of plain queries/generations) from the inference service highly exercised (\~/openathena/build\_envs and to a lesser extent \~/openathena/collect\_rollouts have lots of tooling around using the inference engine, especially agentically)

For each capability we want to use GLM 5.3 to generate 10 task proposals. These proposals should be detailed and can require millions of tokens of further work to actually build—detail doesn’t necessarily mean specifying the exact task itself, it may be some pipeline for finding relevant artifacts/researching more to build a task. In general, I’m giving you a wide level of discretion in deciding what a proposal looks like to build the best tasks.  

The tasks may live inside a few varying levels of environment complexity: a pure reasoning task where we have an answer we extract directly at the end of the task and run a grader over; a task that is “lightly agentic” and can built using [https://github.com/rjpower/shellsim](https://github.com/rjpower/shellsim); a fully agentic task where we’re building the task using Docker/Daytona/Harbor (there should be examples of this in build\_envs)

Task verification can be of 3 basic types: simple verifiable (multiple choice question answering, answer in a box); code verifiable (unit tests, integration tests, behavior tests); LLM-as-judge (which uses a rubric to verify)

We want diversity over task type, environment complexity, and verification type, though not all capabilities make sense for all environment types and verification types. We may need an extra GLM5.3 pass to verify/repair proposals if the base proposal generation doesn’t work well in terms of quality/coverage. GLM5.3 is also very good for mass data labeling/reasoning, so using it in the proposal stage isn’t difficult at all. 

After we have high quality proposals for each capability, we’ll want to synthesize each task, one task per GLM5.3 agentic session (or perhaps many sessions if what the first session determines is that this needs to be a multi-session generation)

A task being complicated to make realistic is not an issue at all–it’s perfectly fine if we end up needing a team of agents to build a task, for instance:

* We may need to find a GitHub repository using web search that simulates the task well/the task would live well inside  
* We may need to find documents or information on the web to find e.g. PDFs to fill the environment or as seed data that GLM agents will use to generate the tasks (build\_envs/stage0 contains tooling for web search for the GLM agents–though you have web search of course too)  
* We may need to do web searches to research resources that will help for making the task realistic/based on real workflows (for instance textbooks, lectures, trainings, problem sets, datasets, etc.)  
* GLM5.3 (or multiple agents working in stages) may need to build a highly complex simulation of a piece of software or a situation for a task

The first step is making a recipe–select a subset of the capabilities and build a pipeline using interactive inference priority and batch CoreWeave/Orion capacity to figure out the prompting and the tooling to get everything working

High level/standing notes:

* Our capacity for inference is large and our compute capacity on the iris/coreweave side is too. Outside of small pilot tests, anything running with anything less than *hundreds*\-wide concurrency is per se suspect and requires an explicit reason with what would break by not pushing concurrency further. In addition, what would break cannot be justified by assumption or reasoning–you must actually observe the failure. It is *far* worse to underutilize our generous capacity which we only have for a limited amount of time than preemptively throttle  
* It is important not to underestimate the capabilities of GLM5.3–it can be handed complex arbitrary work and complete it at an extremely high level of quality   
* [https://github.com/marin-community/marin/pull/9187/](https://github.com/marin-community/marin/pull/9187/) contains the spec for task specification we should use for the final generated tasks  
* Realism and quality is more important than coverage–if something doesn’t make sense, make attempts at iterating to fix it but “null” or “no” or “didn’t work” is always a better answer than making something up  
* Use subagents to hand off work aggressively–especially GPT Sol subagents. Writing code for building infra, running a data labeling/generation process, and most other gutwork can be parallelized with subagents while you focus on the core iteration work. Especially using Terra subagents as a first pass to comprehensively investigate all the other work that’s been done on env generation is probably a good idea  
* I’m giving you wide latitude in deciding what each step looks like and/or adding or removing steps as needed—I want the best recipe for building tasks for the capabilities, not necessarily the recipe I’ve sketched