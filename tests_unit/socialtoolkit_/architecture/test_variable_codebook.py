"""
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
"""
# NOTE: Current test function count: 47
from unittest.mock import MagicMock
from pathlib import Path

import networkx as nx
import pytest


from tests_unit.socialtoolkit_.architecture.conftest import (
    variable_codebook_fixture,
    mock_llm,
    mock_database,
    mock_logger,
    mock_resources,
    mock_configs,
    FixtureError,
    variable_codebook_cache_enabled_is_true_fixture,
    variable_codebook_load_from_file_is_false_fixture,
)

from custom_nodes.red_ribbon.socialtoolkit.architecture.factory import (
    make_variable_codebook,
)
from custom_nodes.red_ribbon.socialtoolkit.architecture.variable_codebook import (
    VariableCodebook,
    Variable,
    Assumptions,
    BusinessOwnerAssumptions,
    BusinessAssumptions,
    TaxesAssumptions,
    PromptDecisionTree,
)

@pytest.fixture
def expected_prompt_sequence():
    """Fixture providing expected prompt sequence for sales_tax_city variable."""
    expected_order = [
        "Is there a sales tax in the city?",
        "Is the sales tax rate a flat rate or variable rate?",
        "What are the units of the sales tax rate?",
        "Is the sales tax rate inclusive or exclusive?",
        "Are there any special sales tax exemptions applicable?"
    ]
    return expected_order

@pytest.fixture
def valid_inputs():
    """Fixture providing valid inputs for get_prompt_sequence_for_input method."""
    return {
        "sales_tax": "What is the sales tax rate?",
        "property_tax": "What is the property tax assessment?",
        "no_match": "What is the regulation?",
    }

@pytest.fixture
def valid_variable_kwargs():
    """Fixture providing valid kwargs for creating Variable instances."""
    return {
        "label": "Sales Tax City",
        "item_name": "sales_tax_city",
        "description": "The sales tax rate for the city.",
        "units": "percentage",
        "assumptions": {
            "description": "Assumptions for business owners regarding sales tax.",
            "details": {"assumption_1": "Must comply with local laws."}
        },
        "prompt_decision_tree": PromptDecisionTree()
    }

@pytest.fixture
def valid_new_variable_kwargs():
    """Fixture providing valid kwargs for creating Variable instances."""
    return {
        "label": "Employee Healthcare Benefits",
        "item_name": "employee_healthcare_benefits",
        "description": "The healthcare coverage provided to full-time employees.",
        "units": "coverage_type",
        "assumptions": {
            "description": "Assumptions for HR departments regarding employee benefits.",
            "details": {"assumption_1": "Must meet state insurance requirements."}
        },
        "prompt_decision_tree": PromptDecisionTree()
    }


@pytest.fixture
def variable_fixture(valid_variable_kwargs) -> Variable:
    """Fixture providing a Variable instance for testing."""
    try:
        return Variable(**valid_variable_kwargs)
    except Exception as e:
        raise FixtureError(f"Failed to create variable_fixture: {e}") from e

@pytest.fixture
def variable_fixture_new_var(valid_new_variable_kwargs) -> Variable:
    """Fixture providing a Variable instance for testing."""
    try:
        return Variable(**valid_new_variable_kwargs)
    except Exception as e:
        raise FixtureError(f"Failed to create variable_fixture_new_var: {e}") from e

@pytest.fixture
def variable_fixture_new_desc(variable_fixture):
    try:
        variable = variable_fixture.model_copy()
        variable.description = "Updated description for testing."
        return variable
    except Exception as e:
        raise FixtureError(f"Failed to create variable_fixture_new_desc: {e}") from e

@pytest.fixture
def variable_fixture_update_income(variable_fixture):
    try:
        variable = variable_fixture.model_copy()
        variable.label = "Updated Income Tax"
        return variable
    except Exception as e:
        raise FixtureError(f"Failed to create variable_fixture_update_income: {e}") from e

@pytest.fixture
def variable_fixture_nonexistent_var(variable_fixture):
    try:
        variable = variable_fixture.model_copy()
        variable.item_name = "nonexistent_var"
        return variable
    except Exception as e:
        raise FixtureError(f"Failed to create variable_fixture_new_desc: {e}") from e


@pytest.fixture
def valid_args():
    """Fixture providing valid positional arguments for VariableCodebook methods."""
    return {
        "get_variable": ("get_variable",),
        "get_prompt_sequence": ("get_prompt_sequence",),
        "get_assumptions": ("get_assumptions",),
        "add_variable": ("add_variable",),
        "update_variable": ("update_variable",),
        "unknown_action": ("unknown_action",),
    }

@pytest.fixture
def valid_kwargs(
    variable_fixture, 
    variable_fixture_new_var,
    variable_fixture_new_desc,
    variable_fixture_update_income
    ):
    """Fixture providing valid keyword arguments for VariableCodebook methods."""
    return {
        "get_variable": {"variable_name": "sales_tax_city"},
        "get_variable_rate": {"variable_name": "sales_tax_rate"},
        "get_variable_property_tax": {"variable_name": "property_tax"},
        "get_variable_income_tax": {"variable_name": "income_tax"},
        "get_variable_test": {"variable_name": "test_var"},
        "get_variable_basic": {"variable_name": "basic_var"},
        "get_variable_nonexistent": {"variable_name": "nonexistent_var"},
        "get_variable_missing": {"variable_name": "missing_var"},
        "get_prompt_sequence": {"variable_name": "sales_tax_city"},
        "get_prompt_sequence_basic": {"variable_name": "basic_var"},
        "get_assumptions": {"variable_name": "property_tax"},
        "add_variable_new": {"variable": variable_fixture_new_var},
        "add_variable_duplicate": {"variable": variable_fixture},
        "update_variable": {"variable_name": "sales_tax_city", "variable": variable_fixture_new_desc},
        "update_variable_income": {"variable_name": "income_tax", "variable": variable_fixture_update_income},
        "update_variable_nonexistent": {"variable_name": "nonexistent_var", "variable": variable_fixture},
    }





class TestControlFlowMethodAcceptsActionParameter:
    """
    Tests for VariableCodebook.run() method with various action parameters.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    @pytest.mark.parametrize("action_key,kwargs_key,expected_value", [
        ("get_variable", "get_variable_rate", True),
        ("add_variable", "add_variable_new", True),
        ("update_variable", "update_variable_income", True),
    ]) # NOTE: Done
    def test_when_calling_run_with_action_then_expected_result_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs, action_key, kwargs_key, expected_value):
        """
        Given action parameter and appropriate kwargs
        When I call run with the action
        Then the value of 'success' key matches expected_value
        """
        args, kwargs = valid_args[action_key], valid_kwargs[kwargs_key]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] == expected_value, f"Expected result['success'] to equal '{expected_value}', but got '{result['success']}'"

    # NOTE: Done
    def test_when_calling_run_with_get_variable_action_then_dictionary_with_variable_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_variable" and variable_name is "sales_tax_city"
        When I call run with the action
        Then a dictionary with the variable is returned
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(result, dict), f"Expected result to be dict, but got {type(result).__name__}"


    @pytest.mark.parametrize("action_key,kwargs_key,expected_key", [
        ("get_prompt_sequence", "get_variable_rate", "prompt_sequence"),
        ("get_assumptions", "get_assumptions", "assumptions"),
    ]) # NOTE: Done
    def test_when_calling_run_with_action_then_expected_key_is_retrieved(self, variable_codebook_fixture, valid_args, valid_kwargs, action_key, kwargs_key, expected_key):
        """
        Given action parameter and variable_name
        When I call run with the action
        Then the expected key is retrieved in the result
        """
        args, kwargs = valid_args[action_key], valid_kwargs[kwargs_key]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert expected_key in result, f"Expected '{expected_key}' key in result, but got {list(result.keys())}"

    # NOTE: Done
    def test_when_calling_run_with_get_prompt_sequence_action_then_list_of_prompts_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_prompt_sequence", variable_name is "sales_tax_city"
        When I call run with the action
        Then a list of prompts is returned
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(result['prompt_sequence'], list), f"Expected result['prompt_sequence'] to be list, but got {type(result['prompt_sequence'])}"

    # NOTE: Done
    @pytest.mark.parametrize("idx", [0, 1, 2, 3])
    def test_when_calling_run_with_get_prompt_sequence_action_then_list_of_prompts_is_returned(self, idx, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_prompt_sequence", variable_name is "sales_tax_city"
        When I call run with the action
        Then the prompts are strings
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        prompt = result['prompt_sequence'][idx]
        assert isinstance(prompt, str), f"Expected prompt '{idx}' to be str, but got {type(prompt).__name__}"

    # NOTE: Done
    def test_when_calling_run_with_add_variable_action_then_variable_is_added(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "add_variable" and a new Variable object
        When I call run with the action
        Then the variable is added to the codebook
        """
        new_var = valid_kwargs["add_variable_new"]["item_name"]
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert new_var in variable_codebook_fixture.variables, f"Expected {new_var} to be in variables, but got {list(variable_codebook_fixture.variables.keys())}"

    # NOTE: Done
    def test_when_calling_run_with_update_variable_action_then_variable_is_updated(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "update_variable" and variable_name is "income_tax"
        When I call run with the action
        Then the variable is updated in the codebook
        """
        var_key = "income_tax"
        updated_variable = valid_kwargs["update_variable_income"]["variable"]["label"]
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_income"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        actual_label = variable_codebook_fixture.variables[var_key].label
        assert actual_label == updated_variable, f"Expected variables['{var_key}'] to be {updated_variable}, but got {actual_label}"



class TestControlFlowValidatesActionParameter:
    """
    Tests for VariableCodebook.run() method action parameter validation.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    # NOTE: Done
    def test_when_calling_run_with_unknown_action_then_value_error_is_raised(self, variable_codebook_fixture, valid_args):
        """
        Given action parameter is "unknown_action"
        When I call run with the invalid action
        Then a ValueError is raised with message "Unknown action"
        """
        args = valid_args["unknown_action"]
        kwargs = {}
        with pytest.raises(ValueError, match=r"Unknown action"):
            variable_codebook_fixture.run(*args, **kwargs)

    # NOTE: Done
    @pytest.mark.parametrize("invalid_type", [None, 420, 6.4, [], {}, ()])
    def test_when_calling_run_with_non_string_action_then_type_error_is_raised(self, invalid_type, variable_codebook_fixture):
        """
        Given action parameter is a non-string type
        When I call run with the invalid action
        Then a TypeError is raised with message "Action must be a string"
        """
        kwargs = {}
        with pytest.raises(TypeError, match=r"Action must be a string"):
            variable_codebook_fixture.run(invalid_type, **kwargs)


class TestControlFlowReturnsDictionarywithOperationResults:
    """
    Tests for VariableCodebook.run() method return value structure.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    # NOTE: Done
    def test_when_get_variable_succeeds_then_result_contains_success_true(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the result contains key "success" with value True
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    # NOTE: Done
    def test_when_get_variable_succeeds_then_result_contains_variable_key(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the result contains key "variable"
        """
        expected_key = 'variable'
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert expected_key in result, f"Expected '{expected_key}' key in result, but got {list(result.keys())}"

    # NOTE: Done
    def test_when_get_variable_succeeds_then_variable_is_variable_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the variable is a Variable object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(result['variable'], Variable), f"Expected result['variable'] to be Variable, but got {type(result['variable'])}"

    # NOTE: Done
    def test_when_get_variable_fails_then_result_contains_success_false(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the result contains key "success" with value False
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is False, f"Expected result['success'] to be False, but got {result['success']}"

    # NOTE: Done
    def test_when_get_variable_fails_then_result_contains_error_key(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the result contains key "error"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        expected_key = 'error'
        assert expected_key in result, f"Expected 'error' key in result, but got {list(result.keys())}"

    # NOTE: Done
    def test_when_get_variable_fails_then_error_indicates_variable_not_found(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the error indicates "Variable not found"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        expected_msg = "variable not found"
        actual_msg = result['error'].lower()
        assert expected_msg in actual_msg, f"Expected 'variable not found' in error message, but got {actual_msg}"


class TestVariableStructureContainsRequiredFields:
    """
    Tests for Variable structure and required fields returned by VariableCodebook.run().
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    @pytest.mark.parametrize("field_name,kwargs_key", [
        ("label", "get_variable_property_tax"),
        ("item_name", "get_variable_property_tax"),
        ("description", "get_variable_property_tax"),
        ("units", "get_variable_property_tax"),
        ("assumptions", "get_variable_property_tax"),
        ("prompt_decision_tree", "get_variable_income_tax"),
    ]) # NOTE: Done
    def test_when_variable_is_retrieved_then_it_has_required_field(self, variable_codebook_fixture, valid_args, valid_kwargs, field_name, kwargs_key):
        """
        Given a variable exists
        When the variable is retrieved
        Then the variable has the required field
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs[kwargs_key]
        result = variable_codebook_fixture.run(*args, **kwargs)
        variable = result["variable"]
        assert hasattr(variable, field_name), f"Expected variable to have '{field_name}' attribute, but attributes are {dir(variable)}"

    # NOTE: Done
    def test_when_variable_has_assumptions_then_it_is_assumptions_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "property_tax" exists
        When the variable is retrieved
        Then assumptions is an Assumptions object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_property_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        variable = result["variable"]["assumptions"]
        assert isinstance(variable.assumptions, Assumptions), f"Expected variable.assumptions to be Assumptions, but got {type(variable.assumptions)}"

    # NOTE: Done
    def test_when_variable_has_prompt_decision_tree_then_it_is_digraph_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "property_tax" exists
        When the variable is retrieved
        Then prompt_decision_tree is a DiGraph object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_property_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"].prompt_decision_tree.tree
        assert isinstance(tree, PromptDecisionTree), f"Expected tree to be PromptDecisionTree, but got {type(tree).__name__}"


class TestAssumptionsObjectContainsStructuredAssumptionData:
    """
    Tests for Assumptions object structure validation.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    @pytest.mark.parametrize("field_name,field_type,kwargs_key", [
        ("general_assumptions", list, "get_variable_test"),
        ("specific_assumptions", dict, "get_variable_test"),
        ("business_owner", BusinessOwnerAssumptions, "get_variable_test"),
        ("business", BusinessAssumptions, "get_variable_test"),
        ("taxes", TaxesAssumptions, "get_variable_test"),
    ]) # NOTE: Done
    def test_when_assumptions_are_retrieved_then_has_expected_field_and_type(self, field_name, field_type, kwargs_key, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with assumptions
        When assumptions are retrieved
        Then assumptions has the expected field with the correct type
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs[kwargs_key]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assumptions = result["variable"]["assumptions"]
        field_value = getattr(assumptions, field_name)
        assert isinstance(field_value, field_type), f"Expected assumptions.{field_name} to be {field_type.__name__}, but got {type(field_value).__name__}"


    @pytest.mark.parametrize("assumption_type,field_name,kwargs_key", [
        ("business_owner", "annual_gross_income", "get_variable_test"),
        ("business", "year_of_operation", "get_variable_test"),
        ("business", "gross_annual_revenue", "get_variable_test"),
        ("business", "employees", "get_variable_test"),
        ("taxes", "taxes_paid_period", "get_variable_test"),
    ]) # NOTE: Done
    def test_when_assumptions_exist_then_contains_expected_field(self, assumption_type, field_name, kwargs_key, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with assumptions
        When assumptions are retrieved
        Then the assumption type contains the expected field
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs[kwargs_key]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assumptions = result["variable"]["assumptions"]
        assumption_obj = getattr(assumptions, assumption_type)
        assert hasattr(assumption_obj, field_name), f"Expected assumptions.{assumption_type} to have '{field_name}' attribute, but attributes are {dir(assumption_obj)}"


class TestGetPromptSequenceExtractsPromptsfromDecisionTree:
    """
    Tests for prompt sequence extraction from decision trees.
    
    Production method: VariableCodebook.run("get_prompt_sequence", variable_name=...)
    """
    # NOTE: Done
    def test_when_get_prompt_sequence_is_called_then_exactly_n_prompts_are_returned(
            self, expected_prompt_sequence, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" has a prompt decision tree with an arbitrary number of prompts
        When get_prompt_sequence is called
        Then the number of prompts returned equals the number of prompts in the tree
        """
        expected_count = len(expected_prompt_sequence)
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        actual_count = len(result['prompt_sequence'])
        assert actual_count == expected_count, f"Expected {expected_count} prompts, but got {actual_count}"

    # NOTE: Done
    def test_when_get_prompt_sequence_is_called_then_prompts_are_in_tree_traversal_order(
            self, expected_prompt_sequence, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" has a prompt decision tree with prompts
        When get_prompt_sequence is called
        Then prompts are in the same order as in tree traversal
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        prompts = result['prompt_sequence']
        assert prompts == expected_prompt_sequence, f"Expected prompts to be '{expected_prompt_sequence}', but got '{prompts}'"

    # NOTE: Done
    def test_when_variable_has_no_decision_tree_then_result_indicates_failure(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "basic_var" has no prompt decision tree
        When get_prompt_sequence is called
        Then the result indicates failure
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence_basic"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        success = result['success']
        assert success is False, f"Expected result['success'] to be False, but got '{success}'"

    # NOTE: Done
    def test_when_variable_has_no_decision_tree_then_error_message_states_no_tree_found(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "basic_var" has no prompt decision tree
        When get_prompt_sequence is called
        Then error message states "No prompt decision tree found"
        """
        expected_msg = "No prompt decision tree found"
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence_basic"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        
        actual_msg = result['error']
        assert expected_msg in actual_msg, f"Expected 'No prompt decision tree found' in error, but got '{actual_msg}'"


class TestGetPromptSequenceforInputExtractsVariableName:
    """
    Tests for variable name extraction from input and prompt sequence generation.
    
    Production method: VariableCodebook.get_prompt_sequence_for_input(input_data_point)
    """
    # NOTE: Done
    def test_when_input_is_valid_then_returns_list_of_prompts(valid_inputs, variable_codebook_fixture):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then the return is a list
        """
        input_data = valid_inputs["sales_tax"]
        prompts = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert isinstance(prompts, list), f"Expected prompts to be a list, but got {type(prompts).__name__}"

    # NOTE: Done
    def test_when_input_is_valid_then_returns_list_of_prompts(valid_inputs, variable_codebook_fixture):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then list matches expected length
        """
        expected_length = 4
        input_data = valid_inputs["sales_tax"]
        prompts = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        len_prompts = len(prompts)
        assert len_prompts == expected_length, f"Expected prompts to be a length '{expected_length}', but got '{len_prompts}'"

    # NOTE: Done
    def test_when_input_is_about_sales_tax_then_variable_name_is_extracted(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then the variable_name "sales_tax_city" is extracted
        """
        expected_string = "sales tax"
        input_data = valid_inputs["sales_tax"]
        prompts = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert expected_string in prompts, f"Expected {expected_string} to be prompts, but got {prompts}"

    @pytest.mark.parametrize("input_key,expected_min_length", [
        ("sales_tax", 0),
        ("property_tax", 0),
    ]) # NOTE: Done
    def test_when_input_is_about_tax_then_prompts_are_returned(self, variable_codebook_fixture, valid_inputs, input_key, expected_min_length):
        """
        Given input_data_point is about tax
        When get_prompt_sequence_for_input is called
        Then prompts are returned with expected minimum length
        """
        input_data = valid_inputs[input_key]
        prompts = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        len_prompts = len(prompts)
        assert len_prompts > expected_min_length, f"Expected prompts length to be > {expected_min_length}, but got {len_prompts}"

    # NOTE: Done
    def test_when_input_is_about_property_tax_then_variable_name_is_extracted(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the property tax assessment?"
        When get_prompt_sequence_for_input is called
        Then the property taxes are present in the prompts
        """
        expected_string = "property tax"
        input_data = valid_inputs["property_tax"]
        prompts = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert expected_string in prompts, f"Expected {expected_string} to be prompts, but got {prompts}"

    # NOTE: Done
    def test_when_input_has_no_matching_keywords_then_default_variable_is_used(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is a query with no matching variables
        When get_prompt_sequence_for_input is called
        Then return an empty prompt sequence
        """
        zero = 0
        input_data = valid_inputs["no_match"]
        prompts = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        len_prompts = len(prompts)
        assert len_prompts == zero, f"Expected prompts to equal {zero}, but got {len_prompts}"


class TestAddVariablePersistsNewVariabletoCodebook:
    """
    Tests for adding new variables to the codebook.
    
    Production method: VariableCodebook.run("add_variable", variable=...)
    """
    # NOTE: Done
    def test_when_add_variable_is_executed_then_success_status_is_true(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then success status is returned as True
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    # NOTE: Done
    def test_when_add_variable_is_executed_then_variable_is_added_to_variables_dictionary(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then the variable is added to self.variables dictionary
        """
        var_name = valid_kwargs["add_variable_new"]["item_name"]
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        vars = variable_codebook_fixture.variables
        assert var_name in vars, f"Expected '{var_name}' to be in variables, but got {list(vars.keys())}"


    # NOTE: Done
    # TODO: This should be confirmable without resorting to checking the mock call history
    @pytest.mark.parametrize("action_key,kwargs_key,storage_method", [
        ("add_variable", "add_variable_new", "save"),
        ("update_variable", "update_variable", "save"),
    ])
    def test_when_variable_operation_is_executed_then_variable_is_persisted_to_storage(
            self, storage_service, variable_codebook_fixture, valid_args, valid_kwargs, action_key, kwargs_key, storage_method):
        """
        Given a Variable object operation (add or update)
        When the operation is executed with persistence enabled
        Then the variable is persisted to storage
        """
        args, kwargs = valid_args[action_key], valid_kwargs[kwargs_key]
        result = variable_codebook_fixture.run(*args, **kwargs)
        storage_method_mock = getattr(variable_codebook_fixture.storage_service, storage_method)
        assert storage_method_mock.called, f"Expected storage_service.{storage_method} to be called, but called={storage_method_mock.called}"


    # NOTE: Done
    def test_when_adding_duplicate_variable_then_error_key_in_result(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "existing_var" that already exists in the codebook
        When add_variable action is executed
        Then an error is in returned result
        """
        expected_key = 'error'
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_duplicate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert expected_key in result, f"Expected '{expected_key}' key in result, but got {list(result.keys())}"


    # NOTE: Done
    def test_when_adding_duplicate_variable_then_error_indicates_duplicate(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "existing_var" that already exists in the codebook
        When add_variable action is executed
        Then the error indicates duplicate variable
        """
        expected_msg = "already present in codebook"
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_duplicate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        error = result["error"]
        assert expected_msg in error, f"Expected '{error}' in error message, but got {error}"


class TestUpdateVariableModifiesExistingVariable:
    """
    Tests for updating existing variables in the codebook.
    
    Production method: VariableCodebook.run("update_variable", variable_name=..., variable=...)
    """
    # NOTE: Done
    def test_when_update_variable_is_executed_then_success_status_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then success status is returned
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    # NOTE: Done
    def test_when_update_variable_is_executed_then_variable_in_self_variables_is_updated(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then the variable in self.variables is updated
        """
        var_name = "sales_tax_city"
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        updated_var = valid_kwargs["update_variable"]["variable"]["description"]
        assert variable_codebook_fixture.variables[var_name] == updated_var, f"Expected variables['{var_name}'] to be {updated_var}, but got {variable_codebook_fixture.variables['sales_tax_city']}"

    # NOTE: Done
    def test_when_updating_non_existent_variable_then_error_is_raised_or_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable_name "nonexistent_var" does not exist
        When update_variable action is executed
        Then an error is raised or returned
        """
        expected_key = 'error'
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert expected_key in result, f"Expected '{expected_key}' key in result, but got {list(result.keys())}"

    # NOTE: Done
    def test_when_updating_non_existent_variable_then_error_indicates_variable_not_found(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable_name "nonexistent_var" does not exist
        When update_variable action is executed
        Then the error indicates variable not found
        """
        expected_msg = 'variable not found'
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        error_msg = result['error']
        assert expected_msg in error_msg, f"Expected '{expected_msg}' in error message, but got {error_msg}"


class TestVariablesAreLoadedfromFileWhenConfigured:
    """
    Tests for variable loading from file during initialization.
    
    Production method: make_variable_codebook()
    """
    # NOTE: Done
    def test_when_load_from_file_is_true_then_variables_are_loaded_from_file(
            self, mock_resources, mock_configs_with_variable_file):
        """
        Given load_from_file is configured as True and a variables file exists
        When make_variable_codebook is called
        Then variables are loaded from the file
        """
        one = 1
        new_codebook_instance = make_variable_codebook(resources=mock_resources, configs=mock_configs_with_variable_file)
        vars = new_codebook_instance.variables
        assert len(vars) == one, f"Expected variables length to equal {one}, but got {len(vars)}"

    # NOTE: Done
    def test_when_variables_are_loaded_then_self_variables_contains_loaded_variable_keys(self, mock_resources, mock_configs_with_variable_file):
        """
        Given load_from_file is configured as True and a variables file exists
        When make_variable_codebook is called
        Then self.variables contains the keys for the loaded variables
        """
        expected_var = "sales_tax_city"
        new_codebook_instance = make_variable_codebook(resources=mock_resources, configs=mock_configs_with_variable_file)
        actual_vars = new_codebook_instance.variables
        assert expected_var in actual_vars, f"Expected '{expected_var}' to be in variables, but got {list(actual_vars.keys())}"

    # NOTE: Done
    def test_when_variables_are_loaded_then_self_variables_values_are_variable_obj(self, mock_resources, mock_configs_with_variable_file):
        """
        Given load_from_file is configured as True and a variables file exists
        When make_variable_codebook is called
        Then the values in self.variables are Variable objects
        """
        expected_var = "sales_tax_city"
        new_codebook_instance = make_variable_codebook(resources=mock_resources, configs=mock_configs_with_variable_file)
        actual_vars = new_codebook_instance.variables
        var_obj = actual_vars[expected_var]
        assert isinstance(var_obj, Variable), f"Expected variables['{expected_var}'] to be Variable, but got {type(var_obj).__name__}"

    # NOTE: Done
    def test_when_load_from_file_is_false_then_self_variables_starts_empty(self, mock_resources, mock_configs_load_from_file_is_false):
        """
        Given load_from_file is configured as False
        When VariableCodebook is initialized
        Then self.variables does not contain keys for that variable
        """
        absent_key = "sales_tax_city"
        new_codebook_instance = make_variable_codebook(resources=mock_resources, configs=mock_configs_load_from_file_is_false)
        actual_vars = new_codebook_instance.variables
        assert absent_key not in actual_vars, f"Expected '{absent_key}' to not be in variables, but got {list(actual_vars.keys())}"


class TestPromptDecisionTreeIsRepresentedasDiGraph:
    """
    Tests for prompt decision tree structure and serialization.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    # NOTE: Done
    def test_when_decision_tree_is_accessed_then_it_is_nx_digraph_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with a prompt decision tree
        When the decision tree is accessed
        Then it is a PromptDecisionTree object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert isinstance(tree, PromptDecisionTree), f"Expected decision_tree to be PromptDecisionTree, but got {type(tree).__name__}"

    # NOTE: Done
    def test_when_decision_tree_is_accessed_then_it_contains_nodes_and_edges(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with a prompt decision tree
        When the decision tree is accessed
        Then it contains nodes and edges
        """
        zero = 0
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert len(tree.nodes) > 0, f"Expected decision_tree.nodes length to be > 0, but got {len(tree.nodes)}"

    @pytest.mark.parametrize("idx", [1, 2, 3, 4, 5]) # NOTE: Done
    def test_when_tree_nodes_are_examined_then_each_node_has_prompt_attribute(self, idx, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a decision tree for a variable
        When tree nodes are examined
        Then each node has a "prompt" attribute
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        node = result["variable"]["prompt_decision_tree"]["nodes"][idx]
        assert hasattr(node, 'prompt'), f"Expected node {idx} to have 'prompt' attribute, but attributes are {dir(node)}"

    @pytest.mark.parametrize("idx", [1, 2, 3, 4, 5]) # NOTE: Done
    def test_when_tree_nodes_are_examined_then_prompts_are_strings(self, idx, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a decision tree for a variable
        When tree nodes are examined
        Then prompts are strings
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        node = result["variable"]["prompt_decision_tree"]["nodes"][idx]
        assert isinstance(node.prompt, str), f"Expected prompts in node to be str, but got {type(node.prompt).__name__}"

