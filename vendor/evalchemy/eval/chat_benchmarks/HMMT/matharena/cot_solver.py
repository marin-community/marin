from loguru import logger


class CoTSolver:
    def __init__(self, querier, system_prompt: str = None):
        """
        Initializes the CoTSolver with the given parameters.

        Args:
            querier: The querier object used for querying the model.
            system_prompt (str, optional): The system prompt to be used. Defaults to None.
            parse_feedback (bool, optional): Flag to enable parsing feedback. Defaults to False.
            check_feedback (bool, optional): Flag to enable checking feedback. Defaults to False.
        """
        self.querier = querier
        self.cost = 0
        self.detailed_cost = []
        self.system_prompt = system_prompt

    def build_query(self, problem):
        """
        Constructs an initial query message for the given problem.

        Args:
            problem (object): The problem instance to build the query for. It should have a  method and a
                              get_formatting_instructions method.

        Returns:
            list: A list of message dictionaries formatted for the query. The list includes a system prompt if
                  provided, followed by the user prompt containing the problem description and formatting instructions.
        """
        prompt, image_path = str(problem[0]), problem[1]
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages, image_path

    def build_queries(self, problems):
        """
        Build a list of queries from a list of problems.

        Args:
            problems (list): A list of problem instances.

        Returns:
            list: A list of queries generated from the given problems.
        """
        queries = []
        for problem in problems:
            queries.append(self.build_query(problem))
        return queries

    def add_response(self, query, response):
        query, _ = query
        if isinstance(response, tuple) and response[0] is None:
            query.append({"role": "api_error", "content": str(response[1])})
        else:
            query.append({"role": "assistant", "content": response})
        return query

    def solve(self, problems):
        """
        Solves the initial round of problems by building queries, running them, and appending responses.

        Args:
            problems (list): A list of problems to be solved.

        Returns:
            list: A list of queries with appended responses from the assistant.
        """
        logger.info("Solving problems.")
        queries = self.build_queries(problems)
        self.cost = 0
        self.detailed_cost = [
            {
                "cost": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            for _ in range(len(problems))
        ]
        for idx, response, detailed_cost in self.querier.run_queries(queries):
            messages = self.add_response(queries[idx], response)
            self.detailed_cost[idx]["cost"] += detailed_cost["cost"]
            self.detailed_cost[idx]["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost[idx]["output_tokens"] += detailed_cost["output_tokens"]
            yield idx, messages, detailed_cost
