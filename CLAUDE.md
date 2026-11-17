# Claude Code Project Instructions

## Project Description
- Pyrometheus is a code generator for reacting-flow thermochemistry.
- It can generate Python, C++, and Fortran code.
- It uses a symbolic representation, based on the Pymbolic library, from which general-purpose code is generated.
- Symbolic expressions are populated with mechanism-specific parameters by invoking a library that acts as an interface to thermochemical data. 

## Non-negotiables
- Follow test-driven development. For every new feature, develop a test, then the code that satisfies it.
- Preserve existing public APIs unless explicitly told.
- Preserve and make use of code structures (`Thermochemistry`, `PythonCodeGenerator`) unless explicitly told.
- Prefer small, verifiable commits/patches (one concept per change).
- Loops:
  - You can use list comprehension when assembling expressions
  - The generated code is intended to be array-based if the language allows for it
    * This is certainly true of Python, where the generated code assumes an array library `pyro_np` will take the burden of prescribing array semantics.
    * The same rule can be respected with C++ through templating.
    * Generated Fortran embeds the calculation in loops, offloaded with directive-based approaches.
  - Preserve dependencies and base classes that make this sub-project work.

## Code style
- Naming: `mass_fractions` for variables/functions, `Thermochemistry` for types.
- No upper-case letters in variables and functions. Never!
- Separate letters and numbers in variable names by an underscore if possible but...
- Avoid numbers in function, class names.
  - For example, `nasa7` and `nasa9` are allowed.
  - In contrast, inexpressive names like `param0`, `Reaction1` are forbidden.
- Explicit, meaningful names such as `concentration`, `temperature`, `Variable`.
- Keep functions < 80 lines unless there is a strong reason.

## Array programming for Python
- Generated code must flexibly works with multiple array libraries: NumPy, JAX, and CuPyNumeric for now.
- The generated code must a `pyro_np` namespace as an input. There is no further need to constrain this input. Distinctions are to be implemented by the user.

## Numerics
- Fow now, Pyrometheus is not meant to produce code that implements numerical methods; the focus is thermochemistry expressions.
- Pyrometheus-generated code must have low entry barriers to work with numerical codes.
- In Python, for the most part, the compatibility burden falls on the array library.

## Testing expectations
- Follow test-driven development. Write a test first, then the code that satisfies it.
- Generated code has to be tested against the source library.
- If you can’t test, explain what would need to be tested and why.

## Interaction rules
- If requirements are ambiguous, choose the least invasive option and leave a TODO.
- When editing, include a brief comment explaining the rationale for any non-obvious change.

## Reversibility rules (non-negotiable)
- Create a new branch immediately, unless told otherwise.
- Use expressive branch names such as `template-free` or `real-gas-eos`
  - Never start a branch with `feature/`
- Make changes in small commits/patches (one logical change each).
- Preserve current behavior at every step; add tests before changing defaults.
- Keep old APIs until requested changes are complete.
- Do not mix formatting-only changes with logic changes.
- For any change that touches generated code, add a unit test or parity check.
