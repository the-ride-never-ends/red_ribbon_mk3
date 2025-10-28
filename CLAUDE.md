# Red Ribbon Development Guide

## General Rules
- Write down the results of your actions in the "Diary" section.
- Diary is for writing the results of your actions over time. This is to keep your memory persistant over instances and to keep your context window as clear as possible.
- If you run into problems with Python already running when trying to start ComfyUI, use the command "pkill -f main.py".
- Your working directory for this project is `custom_nodes/red_ribbon`.

## Testing Red Ribbon Modules
- Run all Red Ribbon tests: `pytest tests-unit/ -k "red_ribbon"`
- Test specific module: `pytest tests-unit/custom_nodes/red_ribbon/path/to/test.py`
- Test specific class: `pytest tests-unit/custom_nodes/red_ribbon/path/to/test.py::TestClassName`
- Test specific method: `pytest tests-unit/custom_nodes/red_ribbon/path/to/test.py::TestClassName::test_method_name`
- Debug with verbosity: `pytest -vv tests-unit/custom_nodes/red_ribbon/`
- Show print output: `pytest -s tests-unit/custom_nodes/red_ribbon/`

## Red Ribbon Code Style
- Type hints required for all parameters and returns
- Import order: standard library, third-party, local modules
- Class naming: PascalCase (RedRibbon, DatabaseAPI)
- Function/variable naming: snake_case (check_for_missing_attributes)
- Private methods: prefix with underscore (_missing_attributes)
- Error handling: use try/except with specific error types and detailed messages
- Database operations should use context managers

## Red Ribbon Architecture
- Modular design with clear separation of concerns
- Class-based with explicit inheritance patterns
- Node-based architecture follows ComfyUI patterns
- Resources should be properly initialized in __init__ methods
- Config-driven design using configs.yaml and private_configs.yaml

## Diary
- **2025-03-01 2:09 PM**: Debugging server initialization issues:
  - Fixed dictionary modification during iteration in `instantiate()` by creating a copy of dict before iteration
  - Fixed import path issues by properly constructing module paths relative to package
  - Added centralized logger module with `make_logger()` function to fix LLM initialization
  - Created test folder structure for Red Ribbon modules
  - Remaining issues: Resource loading for SocialToolkit components needs further debugging
  - Killing processes from terminal

- **2025-10-21**: Completed comprehensive Gherkin documentation for SocialToolkit:
  - Created 10 feature files (2,015 lines) documenting all architecture and key resource components
  - 264 scenarios covering externally verifiable behaviors with concrete examples
  - 113 behavioral rules organized for clarity
  - All files follow Given-When-Then pattern with specific data formats and validation rules
  - Architecture layer: prompt_decision_tree, top10_document_retrieval, document_retrieval_from_websites, relevance_assessment, document_storage, variable_codebook
  - Resource layer: vector_search_engine, ranking_algorithm, cache_manager, query_processor
  - Documentation covers: API contracts, input/output types, error handling, edge cases, configuration effects, logging
  - Created comprehensive README.md summarizing all documentation
  - All quality checks passed: proper structure, balanced Given-When-Then, concrete examples