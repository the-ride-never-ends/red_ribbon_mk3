Feature: Variable Codebook
  As a data collection system
  I want to manage variable definitions with their assumptions and prompt sequences
  So that data extraction follows consistent variable specifications

  Background:
    Given a VariableCodebook instance is initialized
    And a storage service is available
    And a cache service is available
    And a database connection is available
    And an LLM API client is available

  Rule: Control Flow Method Accepts Action Parameter

    Scenario: Control flow with "get_variable" action
      Given action parameter is "get_variable"
      And variable_name is "sales_tax_city"
      When I call run with the action
      Then the variable is retrieved
      And a dictionary with the variable is returned

    Scenario: Control flow with "get_prompt_sequence" action
      Given action parameter is "get_prompt_sequence"
      And variable_name is "sales_tax_city"
      And input_data_point is "What is the sales tax rate?"
      When I call run with the action
      Then the prompt sequence is retrieved
      And a list of prompts is returned

    Scenario: Control flow with "get_assumptions" action
      Given action parameter is "get_assumptions"
      And variable_name is "property_tax"
      When I call run with the action
      Then assumptions for the variable are retrieved

    Scenario: Control flow with "add_variable" action
      Given action parameter is "add_variable"
      And a new Variable object
      When I call run with the action
      Then the variable is added to the codebook
      And success status is returned

    Scenario: Control flow with "update_variable" action
      Given action parameter is "update_variable"
      And variable_name is "income_tax"
      And an updated Variable object
      When I call run with the action
      Then the variable is updated in the codebook
      And success status is returned

  Rule: Control Flow Validates Action Parameter

    Scenario: Control flow rejects unknown action
      Given action parameter is "unknown_action"
      When I call run with the invalid action
      Then a ValueError is raised
      And the error message indicates "Unknown action"

  Rule: Control Flow Returns Dictionary with Operation Results

    Scenario: Successful get_variable returns success flag
      Given a valid variable_name "sales_tax_city"
      When get_variable action is executed
      Then the result contains key "success" with value True
      And the result contains key "variable"
      And the variable is a Variable object

    Scenario: Failed get_variable returns error
      Given an invalid variable_name "nonexistent_var"
      When get_variable action is executed
      Then the result contains key "success" with value False
      And the result contains key "error"
      And the error indicates "Variable not found"

  Rule: Variable Structure Contains Required Fields

    Scenario: Variable has all required fields
      Given a variable "sales_tax_city" exists
      When the variable is retrieved
      Then the variable has field "label"
      And the variable has field "item_name"
      And the variable has field "description"
      And the variable has field "units"

    Scenario: Variable has optional assumptions field
      Given a variable "property_tax" exists
      When the variable is retrieved
      Then the variable may have field "assumptions"
      And if assumptions exist, it is an Assumptions object

    Scenario: Variable has optional prompt_decision_tree field
      Given a variable "income_tax" exists
      When the variable is retrieved
      Then the variable may have field "prompt_decision_tree"
      And if prompt_decision_tree exists, it is a DiGraph object

  Rule: Assumptions Object Contains Structured Assumption Data

    Scenario: Assumptions has general assumptions list
      Given a variable with assumptions
      When assumptions are retrieved
      Then assumptions may contain "general_assumptions" as a list of strings

    Scenario: Assumptions has specific assumptions dictionary
      Given a variable with assumptions
      When assumptions are retrieved
      Then assumptions may contain "specific_assumptions" as a dictionary

    Scenario: Assumptions has business owner assumptions
      Given a variable with business owner assumptions
      When assumptions are retrieved
      Then assumptions may contain "business_owner" field
      And business_owner contains "has_annual_gross_income"

    Scenario: Assumptions has business assumptions
      Given a variable with business assumptions
      When assumptions are retrieved
      Then assumptions may contain "business" field
      And business contains fields like "year_of_operation", "gross_annual_revenue", "employees"

    Scenario: Assumptions has tax assumptions
      Given a variable with tax assumptions
      When assumptions are retrieved
      Then assumptions may contain "taxes" field
      And taxes contains "taxes_paid_period"

  Rule: Get Prompt Sequence Extracts Prompts from Decision Tree

    Scenario: Prompt sequence is extracted from decision tree
      Given variable "sales_tax_city" has a prompt decision tree
      And the tree contains 3 prompts
      When get_prompt_sequence is called
      Then exactly 3 prompts are returned
      And prompts are in tree traversal order

    Scenario: Variable without decision tree returns error
      Given variable "basic_var" has no prompt decision tree
      When get_prompt_sequence is called
      Then the result indicates failure
      And error message states "No prompt decision tree found"

  Rule: Get Prompt Sequence for Input Extracts Variable Name

    Scenario: Input about sales tax maps to sales_tax_city variable
      Given input_data_point is "What is the sales tax rate?"
      When get_prompt_sequence_for_input is called
      Then the variable_name "sales_tax_city" is extracted
      And prompts for sales_tax_city are returned

    Scenario: Input about property tax maps to property_tax variable
      Given input_data_point is "What is the property tax assessment?"
      When get_prompt_sequence_for_input is called
      Then the variable_name "property_tax" is extracted
      And prompts for property_tax are returned

    Scenario: Input with no matching keywords uses default variable
      Given input_data_point is "What is the regulation?"
      And no keywords match existing variables
      When get_prompt_sequence_for_input is called
      Then the default variable "generic_tax_information" is used

  Rule: Add Variable Persists New Variable to Codebook

    Scenario: New variable is added successfully
      Given a Variable object with label "new_var"
      And "new_var" does not exist in the codebook
      When add_variable action is executed
      Then the variable is added to self.variables dictionary
      And the variable is persisted to storage if configured
      And success status is returned

    Scenario: Adding duplicate variable is rejected
      Given a Variable object with label "existing_var"
      And "existing_var" already exists in the codebook
      When add_variable action is executed
      Then an error is raised or returned
      And the error indicates duplicate variable

  Rule: Update Variable Modifies Existing Variable

    Scenario: Existing variable is updated successfully
      Given variable "sales_tax_city" exists
      And an updated Variable object with new description
      When update_variable action is executed
      Then the variable in self.variables is updated
      And the updated variable is persisted to storage
      And success status is returned

    Scenario: Updating non-existent variable fails
      Given variable_name "nonexistent_var"
      And "nonexistent_var" does not exist
      When update_variable action is executed
      Then an error is raised or returned
      And the error indicates variable not found

  Rule: Variables Are Loaded from File When Configured

    Scenario: Variables are loaded on initialization
      Given load_from_file is configured as True
      And a variables file exists at variables_path
      When VariableCodebook is initialized
      Then variables are loaded from the file
      And self.variables contains the loaded variables

    Scenario: Variable loading is skipped when disabled
      Given load_from_file is configured as False
      When VariableCodebook is initialized
      Then no file loading occurs
      And self.variables starts empty

  Rule: Cache Service Is Used When Enabled

    Scenario: Cache is checked before retrieving variable
      Given cache_enabled is configured as True
      And variable "sales_tax_city" is in cache
      When get_variable is called
      Then the variable is returned from cache
      And storage service is not queried

    Scenario: Cache is populated after storage retrieval
      Given cache_enabled is configured as True
      And variable "property_tax" is not in cache
      When get_variable retrieves from storage
      Then the variable is added to cache
      And cache TTL is set to cache_ttl_seconds

    Scenario: Cache is bypassed when disabled
      Given cache_enabled is configured as False
      When get_variable is called
      Then cache is not checked
      And variable is always retrieved from storage or memory

  Rule: Default Assumptions Are Applied When Enabled

    Scenario: Default assumptions are added to variable
      Given default_assumptions_enabled is configured as True
      And a variable without business assumptions
      When the variable is retrieved
      Then default business assumptions are applied
      And assumptions include default values

    Scenario: Default assumptions are not applied when disabled
      Given default_assumptions_enabled is configured as False
      And a variable without assumptions
      When the variable is retrieved
      Then no default assumptions are added
      And the variable retains its original assumptions

  Rule: Prompt Decision Tree Is Represented as DiGraph

    Scenario: Decision tree is a NetworkX DiGraph
      Given a variable with a prompt decision tree
      When the decision tree is accessed
      Then it is a nx.DiGraph object
      And it contains nodes and edges

    Scenario: Decision tree nodes have prompt attributes
      Given a decision tree for a variable
      When tree nodes are examined
      Then each node has a "prompt" attribute
      And prompts are strings

    Scenario: Decision tree can be serialized and deserialized
      Given a variable with a decision tree
      When the variable is serialized to dict
      Then the decision tree is serialized to dict format
      When the dict is deserialized back to Variable
      Then the decision tree is restored as a DiGraph

  Rule: Keyword Matching Maps Input to Variables

    Scenario Outline: Keyword in input maps to correct variable
      Given input contains keyword "<keyword>"
      When variable is extracted from input
      Then the variable_name is "<variable_name>"

      Examples:
        | keyword       | variable_name   |
        | sales tax     | sales_tax_city  |
        | tax rate      | sales_tax_city  |
        | local tax     | sales_tax_city  |
        | city tax      | sales_tax_city  |
        | municipal tax | sales_tax_city  |
        | property tax  | property_tax    |
        | income tax    | income_tax      |

  Rule: Codebook Operations Are Logged

    Scenario: Control flow logs operation start
      Given any action is executed
      When run is called
      Then a log message indicates "Starting variable codebook operation: <action>"

    Scenario: Variable not found is logged as warning
      Given variable_name "missing_var" does not exist
      When get_variable is called
      Then a warning is logged indicating "Variable not found: missing_var"

    Scenario: Missing decision tree is logged as warning
      Given a variable without prompt decision tree
      When get_prompt_sequence is called
      Then a warning is logged indicating "No prompt decision tree found"
