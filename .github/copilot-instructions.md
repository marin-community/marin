# GitHub Copilot Code Review Instructions

## Primary Goal
Provide **actionable feedback on real bugs and logic errors**. Focus on issues that would cause runtime failures, incorrect behavior, or subtle algorithmic problems.

## What TO Review

### Critical Issues (Always Flag)
- **Logic errors**: Off-by-one errors, incorrect loop conditions, wrong operators
- **Bugs that cause crashes**: None value errors (e.g., AttributeError on None), index out of bounds, type mismatches
- **Data flow errors**: Variables used before initialization, incorrect data transformations
- **Algorithmic correctness**: Mathematical errors, incorrect tensor shapes/dimensions, wrong model behavior
- **Security vulnerabilities**: SQL injection, XSS, command injection, exposed secrets
- **Resource leaks**: Unclosed files/connections (only if not using context managers), memory leaks
- **Race conditions and concurrency bugs**: Deadlocks, data races, improper synchronization
- **API misuse**: Incorrect API usage that will cause failures, mismatched tensor operations

### Important Issues (Flag if Significant)
- **Performance bugs**: O(n²) where O(n) is trivial, unnecessary nested loops causing major slowdowns
- **Silent failures**: Errors being swallowed without proper handling
- **Incorrect error handling**: Catching exceptions too broadly and masking real issues

## What NOT to Review

### Automatically Handled (NEVER Flag)
- **Code formatting**: Line length, spacing, indentation - handled by pre-commit hooks
- **Import ordering**: Import statement organization - handled by formatters
- **Type annotations**: Missing or incorrect type hints (unless causing runtime errors)
- **Linting issues**: Unused imports, unused variables, naming conventions - handled by linters

### Stylistic Preferences (NEVER Flag)
- Import style choices (from X import Y vs import X)
- Variable naming preferences (unless genuinely confusing)
- Comment formatting or style
- Docstring formatting (unless fundamentally incorrect)
- Code organization preferences

### Research/Experimental Code Context (NEVER Flag)
- "Production best practices" like assertions vs explicit raises in experimental scripts
- Unused parameters that are clearly placeholders for future work
- Commented-out code that's left for reference
- Unused imports in exploratory/research code
- Generic advice like "assertions are bad for production" in non-production code

### Minor Technical Debt (NEVER Flag)
- Missing docstrings
- Incomplete error messages
- TODO comments
- Temporary workarounds (unless they introduce bugs)
- Parameter names like `add_special_tokens` being ignored (implementation choices)

## Review Guidelines

### Prioritization
1. **Stop if you find >3 issues**: If there are many minor issues, provide ONLY the 3 most critical ones and stop
2. **One comment per significant issue**: Combine related minor issues into a single comment
3. **Skip nitpicking entirely**: If all issues are minor, provide NO feedback rather than overwhelm the developer

### Comment Quality
- **Be specific**: Point to exact line numbers and explain the impact
- **Provide fixes**: Suggest concrete solutions, not just problems
- **Explain why**: Briefly explain why something is wrong, not just that it is
- **Assume competence**: Don't explain basic programming concepts

### Uncertainty
- **If unsure, skip it**: Only flag issues you're confident about
- **Context matters**: Consider the surrounding code and overall architecture
- **Ask questions**: If something seems wrong but might be intentional, phrase as a question

## Examples of Good vs Bad Feedback

### ✅ GOOD: Actionable Bug
```
The temperature is applied during sampling but not during log probability calculation (lines 45-52).
This creates a policy mismatch that will bias gradients. Apply temperature in both places:
log_probs = jax.nn.log_softmax(logits / temperature, axis=-1)
```

### ✅ GOOD: Logic Error
```
`compute_reward` docstring claims substring matching but implements exact equality (line 78).
This will fail for partial matches. Use: if expected_answer.lower() in generated_text.lower()
```

### ❌ BAD: Import Nitpicking
```
Module 'haliax' is imported with both 'import' and 'from import' statements.
```

### ❌ BAD: Unused Variable in Experimental Code
```
Variable `vocab` is not used anywhere in this function.
```

### ❌ BAD: Generic Best Practice
```
Using assert statements is not recommended for production code. Use explicit if/raise instead.
```

### ❌ BAD: Formatting Issue
```
This comment is poorly formatted with multiple # symbols on the same line.
```

## Final Checklist Before Posting
- [ ] Would this cause a runtime failure or incorrect behavior?
- [ ] Is this something auto-formatters/linters can't catch?
- [ ] Is the fix non-obvious or worth pointing out?
- [ ] Am I providing actionable advice, not just criticism?
- [ ] Is this specific to the code context, not generic advice?

**If you can't check all boxes, don't post the comment.**
