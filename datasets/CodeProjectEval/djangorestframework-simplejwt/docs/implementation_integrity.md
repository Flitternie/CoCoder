# Implementation Integrity Rules

Follow these restrictions while implementing this repository.

## General do not rules

- Do not install, import from, inspect source code from, copy source code from, or delegate to the target package, a same-name package, or a compatibility-equivalent drop-in implementation, even if the package name is different.
- Do not read, import, execute, or copy from the dataset reference implementation outside the assigned workspace.
- Do not modify tests, test data, fixtures, reports, or evaluation scripts.
- Do not hardcode visible test answers, fixture data, filenames, IDs, timestamps, credentials, messages, or expected outputs.
- Do not add test-only branches or logic that depends on unit/check/acceptance test internals.
- Do not replace required behavior with mocks, stubs, no-op placeholders, constant returns, fake files, generic success, or fixture-specific dispatch/response tables.

## Allowed reference boundary

- Public documentation, API references, papers, general algorithm descriptions, and helper libraries are allowed.
- Helper libraries may be used for implementation work, including core sub-tasks, as long as they are not the target package, not a same-name package, and not a compatibility-equivalent drop-in implementation.

## Repository-specific do not rules

- Do not emit dummy, constant, or unsigned JWTs, or skip token validation and authentication semantics.
