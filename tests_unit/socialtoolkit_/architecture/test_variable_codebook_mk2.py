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
import pytest
from tests-unit.socialtoolkit_.architecture.conftest import (
    variable_codebook_fixture
)

# Fixtures for Background

@pytest.fixture
def a_variablecodebook_instance_is_initialized():
    """
    Given a VariableCodebook instance is initialized
    """
    pass


@pytest.fixture
def a_storage_service_is_available():
    """
    And a storage service is available
    """
    pass


@pytest.fixture
def a_cache_service_is_available():
    """
    And a cache service is available
    """
    pass


@pytest.fixture
def a_database_connection_is_available():
    """
    And a database connection is available
    """
    pass


@pytest.fixture
def an_llm_api_client_is_available():
    """
    And an LLM API client is available
    """
    pass


def make_variable_codebook():
    return variable_codebook_fixture


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
def valid_kwargs():
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
        "add_variable_new": {"variable": {"label": "new_var", "item_name": "new_var"}},
        "add_variable_duplicate": {"variable": {"label": "existing_var", "item_name": "existing_var"}},
        "update_variable": {"variable_name": "sales_tax_city", "variable": {"description": "updated description"}},
        "update_variable_income": {"variable_name": "income_tax", "variable": {"label": "updated"}},
        "update_variable_nonexistent": {"variable_name": "nonexistent_var", "variable": {"description": "new description"}},
    }


@pytest.fixture
def valid_inputs():
    """Fixture providing valid inputs for get_prompt_sequence_for_input method."""
    return {
        "sales_tax": "What is the sales tax rate?",
        "property_tax": "What is the property tax assessment?",
        "no_match": "What is the regulation?",
    }


class TestControlFlowMethodAcceptsActionParameter:
    """
    Tests for VariableCodebook.run() method with various action parameters.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_calling_run_with_get_variable_action_then_variable_is_retrieved(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_variable" and variable_name is "sales_tax_city"
        When I call run with the action
        Then the variable is retrieved
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    def test_when_calling_run_with_get_variable_action_then_dictionary_with_variable_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_variable" and variable_name is "sales_tax_city"
        When I call run with the action
        Then a dictionary with the variable is returned
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(result, dict), f"Expected result to be dict, but got {type(result)}"

    def test_when_calling_run_with_get_prompt_sequence_action_then_prompt_sequence_is_retrieved(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_prompt_sequence", variable_name is "sales_tax_city"
        When I call run with the action
        Then the prompt sequence is retrieved
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    def test_when_calling_run_with_get_prompt_sequence_action_then_list_of_prompts_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_prompt_sequence", variable_name is "sales_tax_city"
        When I call run with the action
        Then a list of prompts is returned
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_variable_rate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(result['prompt_sequence'], list), f"Expected result['prompt_sequence'] to be list, but got {type(result['prompt_sequence'])}"

    def test_when_calling_run_with_get_assumptions_action_then_assumptions_are_retrieved(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "get_assumptions" and variable_name is "property_tax"
        When I call run with the action
        Then assumptions for the variable are retrieved
        """
        args, kwargs = valid_args["get_assumptions"], valid_kwargs["get_assumptions"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'assumptions' in result, f"Expected 'assumptions' key in result, but got {list(result.keys())}"

    def test_when_calling_run_with_add_variable_action_then_variable_is_added(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "add_variable" and a new Variable object
        When I call run with the action
        Then the variable is added to the codebook
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert new_variable.item_name in variable_codebook_fixture.variables, f"Expected {new_variable.item_name} to be in variables, but got {list(variable_codebook_fixture.variables.keys())}"

    def test_when_calling_run_with_add_variable_action_then_success_status_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "add_variable" and a new Variable object
        When I call run with the action
        Then success status is returned
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    def test_when_calling_run_with_update_variable_action_then_variable_is_updated(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "update_variable" and variable_name is "income_tax"
        When I call run with the action
        Then the variable is updated in the codebook
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_income"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert variable_codebook_fixture.variables['income_tax'] == updated_variable, f"Expected variables['income_tax'] to be {updated_variable}, but got {variable_codebook_fixture.variables['income_tax']}"

    def test_when_calling_run_with_update_variable_action_then_success_status_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given action parameter is "update_variable" and variable_name is "income_tax"
        When I call run with the action
        Then success status is returned
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_income"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

class TestControlFlowValidatesActionParameter:
    """
    Tests for VariableCodebook.run() method action parameter validation.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_calling_run_with_unknown_action_then_value_error_is_raised(self, variable_codebook_fixture, valid_args):
        """
        Given action parameter is "unknown_action"
        When I call run with the invalid action
        Then a ValueError is raised with message "Unknown action"
        """
        with pytest.raises(ValueError, match="Unknown action"):
            args = valid_args["unknown_action"]
            variable_codebook_fixture.run(*args)


class TestControlFlowReturnsDictionarywithOperationResults:
    """
    Tests for VariableCodebook.run() method return value structure.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_get_variable_succeeds_then_result_contains_success_true(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the result contains key "success" with value True
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    def test_when_get_variable_succeeds_then_result_contains_variable_key(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the result contains key "variable"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'variable' in result, f"Expected 'variable' key in result, but got {list(result.keys())}"

    def test_when_get_variable_succeeds_then_variable_is_variable_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the variable is a Variable object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(result['variable'], Variable), f"Expected result['variable'] to be Variable, but got {type(result['variable'])}"

    def test_when_get_variable_fails_then_result_contains_success_false(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the result contains key "success" with value False
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is False, f"Expected result['success'] to be False, but got {result['success']}"

    def test_when_get_variable_fails_then_result_contains_error_key(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the result contains key "error"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'error' in result, f"Expected 'error' key in result, but got {list(result.keys())}"

    def test_when_get_variable_fails_then_error_indicates_variable_not_found(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the error indicates "Variable not found"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'variable not found' in result['error'].lower(), f"Expected 'variable not found' in error message, but got {result['error']}"


class TestVariableStructureContainsRequiredFields:
    """
    Tests for Variable structure and required fields returned by VariableCodebook.run().
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_variable_is_retrieved_then_it_has_label_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "label"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert hasattr(variable, 'label'), f"Expected variable to have 'label' attribute, but attributes are {dir(variable)}"

    def test_when_variable_is_retrieved_then_it_has_item_name_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "item_name"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert hasattr(variable, 'item_name'), f"Expected variable to have 'item_name' attribute, but attributes are {dir(variable)}"

    def test_when_variable_is_retrieved_then_it_has_description_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "description"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert hasattr(variable, 'description'), f"Expected variable to have 'description' attribute, but attributes are {dir(variable)}"

    def test_when_variable_is_retrieved_then_it_has_units_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "units"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert hasattr(variable, 'units'), f"Expected variable to have 'units' attribute, but attributes are {dir(variable)}"

    def test_when_variable_is_retrieved_then_it_may_have_assumptions_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "property_tax" exists
        When the variable is retrieved
        Then the variable may have field "assumptions"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_property_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert hasattr(variable, 'assumptions'), f"Expected variable to have 'assumptions' attribute, but attributes are {dir(variable)}"

    def test_when_variable_has_assumptions_then_it_is_assumptions_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "property_tax" exists
        When the variable is retrieved
        Then if assumptions exist, it is an Assumptions object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_property_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assumptions = result["variable"]["assumptions"]
        assert isinstance(variable.assumptions, Assumptions), f"Expected variable.assumptions to be Assumptions, but got {type(variable.assumptions)}"

    def test_when_variable_is_retrieved_then_it_may_have_prompt_decision_tree_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "income_tax" exists
        When the variable is retrieved
        Then the variable may have field "prompt_decision_tree"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_income_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert hasattr(variable, 'prompt_decision_tree'), f"Expected variable to have 'prompt_decision_tree' attribute, but attributes are {dir(variable)}"

    def test_when_variable_has_prompt_decision_tree_then_it_is_digraph_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable "income_tax" exists
        When the variable is retrieved
        Then if prompt_decision_tree exists, it is a DiGraph object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_income_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert isinstance(variable.prompt_decision_tree._graph, nx.DiGraph), f"Expected variable.prompt_decision_tree._graph to be nx.DiGraph, but got {type(variable.prompt_decision_tree._graph)}"


class TestAssumptionsObjectContainsStructuredAssumptionData:
    """
    Tests for Assumptions object structure validation.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_assumptions_are_retrieved_then_may_contain_general_assumptions_list(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with assumptions
        When assumptions are retrieved
        Then assumptions may contain "general_assumptions" as a list of strings
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assumptions = result["variable"]["assumptions"]
        general = assumptions["general_assumptions"]
        assert isinstance(assumptions.general_assumptions, list), f"Expected assumptions.general_assumptions to be list, but got {type(assumptions.general_assumptions)}"

    def test_when_assumptions_are_retrieved_then_may_contain_specific_assumptions_dictionary(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with assumptions
        When assumptions are retrieved
        Then assumptions may contain "specific_assumptions" as a dictionary
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assumptions = result["variable"]["assumptions"]
        specific = assumptions["specific_assumptions"]
        assert isinstance(assumptions.specific_assumptions, dict), f"Expected assumptions.specific_assumptions to be dict, but got {type(assumptions.specific_assumptions)}"

    def test_when_assumptions_are_retrieved_then_may_contain_business_owner_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with business owner assumptions
        When assumptions are retrieved
        Then assumptions may contain "business_owner" field
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(assumptions.business_owner, BusinessOwnerAssumptions), f"Expected assumptions.business_owner to be BusinessOwnerAssumptions, but got {type(assumptions.business_owner)}"

    def test_when_business_owner_assumptions_exist_then_contains_has_annual_gross_income(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with business owner assumptions
        When assumptions are retrieved
        Then business_owner contains "has_annual_gross_income"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        business_owner = result["variable"]["assumptions"]["business_owner"]
        assert hasattr(assumptions.business_owner, 'has_annual_gross_income'), f"Expected assumptions.business_owner to have 'has_annual_gross_income' attribute, but attributes are {dir(assumptions.business_owner)}"

    def test_when_assumptions_are_retrieved_then_may_contain_business_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with business assumptions
        When assumptions are retrieved
        Then assumptions may contain "business" field
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(assumptions.business, BusinessAssumptions), f"Expected assumptions.business to be BusinessAssumptions, but got {type(assumptions.business)}"

    def test_when_business_assumptions_exist_then_contains_expected_fields(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given: a variable with business assumptions
        When: assumptions are retrieved
        Then: business contains fields like "year_of_operation", "gross_annual_revenue", "employees"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        business = result["variable"]["assumptions"]["business"]
        expected_fields = ["year_of_operation", "gross_annual_revenue", "employees"]
        assert hasattr(assumptions.business, 'year_of_operation'), f"Expected assumptions.business to have 'year_of_operation' attribute, but attributes are {dir(assumptions.business)}"

    def test_when_assumptions_are_retrieved_then_may_contain_taxes_field(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with tax assumptions
        When assumptions are retrieved
        Then assumptions may contain "taxes" field
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert isinstance(assumptions.taxes, TaxesAssumptions), f"Expected assumptions.taxes to be TaxesAssumptions, but got {type(assumptions.taxes)}"

    def test_when_tax_assumptions_exist_then_contains_taxes_paid_period(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with tax assumptions
        When assumptions are retrieved
        Then taxes contains "taxes_paid_period"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        taxes = result["variable"]["assumptions"]["taxes"]
        assert hasattr(assumptions.taxes, 'taxes_paid_period'), f"Expected assumptions.taxes to have 'taxes_paid_period' attribute, but attributes are {dir(assumptions.taxes)}"


class TestGetPromptSequenceExtractsPromptsfromDecisionTree:
    """
    Tests for prompt sequence extraction from decision trees.
    
    Production method: VariableCodebook.run("get_prompt_sequence", variable_name=...)
    """
    def test_when_get_prompt_sequence_is_called_then_exactly_n_prompts_are_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" has a prompt decision tree with 3 prompts
        When get_prompt_sequence is called
        Then exactly 3 prompts are returned
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        expected_count = 3
        assert len(prompts) == 3, f"Expected 3 prompts, but got {len(prompts)}"

    def test_when_get_prompt_sequence_is_called_then_prompts_are_in_tree_traversal_order(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" has a prompt decision tree with 3 prompts
        When get_prompt_sequence is called
        Then prompts are in tree traversal order
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert prompts == expected_order, f"Expected prompts to be {expected_order}, but got {prompts}"

    def test_when_variable_has_no_decision_tree_then_result_indicates_failure(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "basic_var" has no prompt decision tree
        When get_prompt_sequence is called
        Then the result indicates failure
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence_basic"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is False, f"Expected result['success'] to be False, but got {result['success']}"

    def test_when_variable_has_no_decision_tree_then_error_message_states_no_tree_found(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "basic_var" has no prompt decision tree
        When get_prompt_sequence is called
        Then error message states "No prompt decision tree found"
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_prompt_sequence_basic"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'No prompt decision tree found' in result['error'], f"Expected 'No prompt decision tree found' in error, but got {result['error']}"


class TestGetPromptSequenceforInputExtractsVariableName:
    """
    Tests for variable name extraction from input and prompt sequence generation.
    
    Production method: VariableCodebook.get_prompt_sequence_for_input(input_data_point)
    """
    def test_when_input_is_about_sales_tax_then_variable_name_is_extracted(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then the variable_name "sales_tax_city" is extracted
        """
        input_data = valid_inputs["sales_tax"]
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert variable_name == 'sales_tax_city', f"Expected variable_name to be 'sales_tax_city', but got {variable_name}"

    def test_when_input_is_about_sales_tax_then_prompts_for_sales_tax_city_are_returned(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then prompts for sales_tax_city are returned
        """
        input_data = valid_inputs["sales_tax"]
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert len(prompts) > 0, f"Expected prompts length to be > 0, but got {len(prompts)}"

    def test_when_input_is_about_property_tax_then_variable_name_is_extracted(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the property tax assessment?"
        When get_prompt_sequence_for_input is called
        Then the variable_name "property_tax" is extracted
        """
        input_data = valid_inputs["property_tax"]
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert variable_name == 'property_tax', f"Expected variable_name to be 'property_tax', but got {variable_name}"

    def test_when_input_is_about_property_tax_then_prompts_for_property_tax_are_returned(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the property tax assessment?"
        When get_prompt_sequence_for_input is called
        Then prompts for property_tax are returned
        """
        input_data = valid_inputs["property_tax"]
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert len(prompts) > 0, f"Expected prompts length to be > 0, but got {len(prompts)}"

    def test_when_input_has_no_matching_keywords_then_default_variable_is_used(self, variable_codebook_fixture, valid_inputs):
        """
        Given input_data_point is "What is the regulation?" with no matching keywords
        When get_prompt_sequence_for_input is called
        Then the default variable "generic_tax_information" is used
        """
        input_data = valid_inputs["no_match"]
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_data)
        assert variable_name == 'generic_tax_information', f"Expected variable_name to be 'generic_tax_information', but got {variable_name}"


class TestAddVariablePersistsNewVariabletoCodebook:
    """
    Tests for adding new variables to the codebook.
    
    Production method: VariableCodebook.run("add_variable", variable=...)
    """
    def test_when_add_variable_is_executed_then_variable_is_added_to_variables_dictionary(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then the variable is added to self.variables dictionary
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'new_var' in variable_codebook_fixture.variables, f"Expected 'new_var' to be in variables, but got {list(variable_codebook_fixture.variables.keys())}"

    def test_when_add_variable_is_executed_then_variable_is_persisted_to_storage(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then the variable is persisted to storage if configured
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert storage_service.save.called, f"Expected storage_service.save to be called, but called={storage_service.save.called}"

    def test_when_add_variable_is_executed_then_success_status_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then success status is returned
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_new"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    def test_when_adding_duplicate_variable_then_error_is_raised_or_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "existing_var" that already exists in the codebook
        When add_variable action is executed
        Then an error is raised or returned
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_duplicate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'error' in result, f"Expected 'error' key in result, but got {list(result.keys())}"

    def test_when_adding_duplicate_variable_then_error_indicates_duplicate(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a Variable object with label "existing_var" that already exists in the codebook
        When add_variable action is executed
        Then the error indicates duplicate variable
        """
        args, kwargs = valid_args["add_variable"], valid_kwargs["add_variable_duplicate"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        error = result.get("error", "").lower()
        assert 'duplicate' in result['error'].lower(), f"Expected 'duplicate' in error message, but got {result['error']}"


class TestUpdateVariableModifiesExistingVariable:
    """
    Tests for updating existing variables in the codebook.
    
    Production method: VariableCodebook.run("update_variable", variable_name=..., variable=...)
    """
    def test_when_update_variable_is_executed_then_variable_in_self_variables_is_updated(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then the variable in self.variables is updated
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert variable_codebook_fixture.variables['sales_tax_city'] == updated_variable, f"Expected variables['sales_tax_city'] to be {updated_variable}, but got {variable_codebook_fixture.variables['sales_tax_city']}"

    def test_when_update_variable_is_executed_then_updated_variable_is_persisted_to_storage(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then the updated variable is persisted to storage
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert storage_service.save.called, f"Expected storage_service.save to be called, but called={storage_service.save.called}"

    def test_when_update_variable_is_executed_then_success_status_is_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then success status is returned
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['success'] is True, f"Expected result['success'] to be True, but got {result['success']}"

    def test_when_updating_non_existent_variable_then_error_is_raised_or_returned(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable_name "nonexistent_var" does not exist
        When update_variable action is executed
        Then an error is raised or returned
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'error' in result, f"Expected 'error' key in result, but got {list(result.keys())}"

    def test_when_updating_non_existent_variable_then_error_indicates_variable_not_found(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given variable_name "nonexistent_var" does not exist
        When update_variable action is executed
        Then the error indicates variable not found
        """
        args, kwargs = valid_args["update_variable"], valid_kwargs["update_variable_nonexistent"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert 'variable not found' in result['error'].lower(), f"Expected 'variable not found' in error message, but got {result['error']}"


class TestVariablesAreLoadedfromFileWhenConfigured:
    """
    Tests for variable loading from file during initialization.
    
    Production method: VariableCodebook.__init__(resources, configs)
    """
    def test_when_load_from_file_is_true_then_variables_are_loaded_from_file(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as True and a variables file exists
        When VariableCodebook is initialized
        Then variables are loaded from the file
        """
        assert len(variable_codebook_fixture.variables) > 0, f"Expected variables length to be > 0, but got {len(variable_codebook_fixture.variables)}"

    def test_when_variables_are_loaded_then_self_variables_contains_loaded_variables(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as True and a variables file exists
        When VariableCodebook is initialized
        Then self.variables contains the loaded variables
        """
        assert len(variable_codebook_fixture.variables) > 0, f"Expected variables length to be > 0, but got {len(variable_codebook_fixture.variables)}"

    def test_when_load_from_file_is_false_then_no_file_loading_occurs(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as False
        When VariableCodebook is initialized
        Then no file loading occurs
        """
        assert not file_read_called, f"Expected file_read_called to be False, but got {file_read_called}"

    def test_when_load_from_file_is_false_then_self_variables_starts_empty(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as False
        When VariableCodebook is initialized
        Then self.variables starts empty
        """
        variable_codebook = 
        assert len(variable_codebook_fixture.variables) == 0, f"Expected variables length to be 0, but got {len(variable_codebook_fixture.variables)}"


class TestCacheServiceIsUsedWhenEnabled:
    """
    Tests for cache service integration.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_cache_is_enabled_and_variable_in_cache_then_variable_is_returned_from_cache(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given cache_enabled is configured as True and variable "sales_tax_city" is in cache
        When get_variable is called
        Then the variable is returned from cache
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert result['variable'] == cached_variable, f"Expected result['variable'] to be {cached_variable}, but got {result['variable']}"

    def test_when_cache_is_enabled_and_variable_in_cache_then_storage_service_is_not_queried(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given cache_enabled is configured as True and variable "sales_tax_city" is in cache
        When get_variable is called
        Then storage service is not queried
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert not storage_service.get.called, f"Expected storage_service.get.called to be False, but got {storage_service.get.called}"

    def test_when_cache_is_enabled_and_variable_not_in_cache_then_variable_is_added_to_cache(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given cache_enabled is configured as True and variable "property_tax" is not in cache
        When get_variable retrieves from storage
        Then the variable is added to cache
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_property_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert cache_service.set.called, f"Expected cache_service.set.called to be True, but got {cache_service.set.called}"

    def test_when_cache_is_enabled_and_variable_cached_then_cache_ttl_is_set(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given cache_enabled is configured as True and variable "property_tax" is not in cache
        When get_variable retrieves from storage
        Then cache TTL is set to cache_ttl_seconds
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_property_tax"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert cache_service.set.call_args[1]['ttl'] == variable_codebook_fixture.configs.cache_ttl_seconds, f"Expected cache TTL to be {variable_codebook_fixture.configs.cache_ttl_seconds}, but got {cache_service.set.call_args[1]['ttl']}"

    def test_when_cache_is_disabled_then_cache_is_not_checked(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given cache_enabled is configured as False
        When get_variable is called
        Then cache is not checked
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert not cache_service.get.called, f"Expected cache_service.get.called to be False, but got {cache_service.get.called}"

    def test_when_cache_is_disabled_then_variable_is_always_retrieved_from_storage(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given cache_enabled is configured as False
        When get_variable is called
        Then variable is always retrieved from storage
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert storage_service.get.called, f"Expected storage_service.get.called to be True, but got {storage_service.get.called}"


class TestDefaultAssumptionsAreAppliedWhenEnabled:
    """
    Tests for default assumptions application.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_default_assumptions_enabled_then_default_business_assumptions_are_applied(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given default_assumptions_enabled is configured as True and a variable without business assumptions
        When the variable is retrieved
        Then default business assumptions are applied
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert variable.assumptions.business is not None, f"Expected variable.assumptions.business to not be None, but got {variable.assumptions.business}"

    def test_when_default_assumptions_enabled_then_assumptions_include_default_values(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given default_assumptions_enabled is configured as True and a variable without business assumptions
        When the variable is retrieved
        Then assumptions include default values
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert variable.assumptions is not None, f"Expected variable.assumptions to not be None, but got {variable.assumptions}"

    def test_when_default_assumptions_disabled_then_no_default_assumptions_are_added(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given default_assumptions_enabled is configured as False and a variable without assumptions
        When the variable is retrieved
        Then no default assumptions are added
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        variable = result["variable"]
        assert variable.assumptions is None, f"Expected variable.assumptions to be None, but got {variable.assumptions}"

    def test_when_default_assumptions_disabled_then_variable_retains_original_assumptions(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given default_assumptions_enabled is configured as False and a variable without assumptions
        When the variable is retrieved
        Then the variable retains its original assumptions
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert variable.assumptions == original_assumptions, f"Expected variable.assumptions to be {original_assumptions}, but got {variable.assumptions}"


class TestPromptDecisionTreeIsRepresentedasDiGraph:
    """
    Tests for prompt decision tree structure and serialization.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_decision_tree_is_accessed_then_it_is_nx_digraph_object(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with a prompt decision tree
        When the decision tree is accessed
        Then it is a nx.DiGraph object
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert isinstance(decision_tree, nx.DiGraph), f"Expected decision_tree to be nx.DiGraph, but got {type(decision_tree)}"

    def test_when_decision_tree_is_accessed_then_it_contains_nodes_and_edges(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable with a prompt decision tree
        When the decision tree is accessed
        Then it contains nodes and edges
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert len(decision_tree.nodes) > 0, f"Expected decision_tree.nodes length to be > 0, but got {len(decision_tree.nodes)}"

    def test_when_tree_nodes_are_examined_then_each_node_has_prompt_attribute(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a decision tree for a variable
        When tree nodes are examined
        Then each node has a "prompt" attribute
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        nodes = result["variable"]["prompt_decision_tree"]["nodes"]
        assert all(hasattr(node, 'prompt') for node in decision_tree.nodes), f"Expected all nodes to have 'prompt' attribute, but some nodes missing it: {[node for node in decision_tree.nodes if not hasattr(node, 'prompt')]}"

    def test_when_tree_nodes_are_examined_then_prompts_are_strings(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a decision tree for a variable
        When tree nodes are examined
        Then prompts are strings
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        nodes = result["variable"]["prompt_decision_tree"]["nodes"]
        assert all(isinstance(node.prompt, str) for node in decision_tree.nodes), f"Expected all prompts to be strings, but got {[type(node.prompt) for node in decision_tree.nodes]}"

    def test_when_variable_is_serialized_then_decision_tree_is_serialized_to_dict(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given: a variable with a decision tree
        When: the variable is serialized to dict and the dict is deserialized back to Variable
        Then: the decision tree is serialized to dict format
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert isinstance(serialized_dict['prompt_decision_tree'], dict), f"Expected serialized_dict['prompt_decision_tree'] to be dict, but got {type(serialized_dict['prompt_decision_tree'])}"

    def test_when_dict_is_deserialized_then_decision_tree_is_restored_as_digraph(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given: a variable with a decision tree
        When: the variable is serialized to dict and the dict is deserialized back to Variable
        Then: the decision tree is restored as a DiGraph
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        tree = result["variable"]["prompt_decision_tree"]
        assert isinstance(deserialized_variable.prompt_decision_tree.graph, nx.DiGraph), f"Expected deserialized decision tree to be nx.DiGraph, but got {type(deserialized_variable.prompt_decision_tree._graph)}"


class TestKeywordMatchingMapsInputtoVariables:
    """
    Tests for keyword matching for variable extraction.
    
    Production method: VariableCodebook.get_prompt_sequence_for_input(input_data_point)
    """
    def test_when_keyword_is_in_input_then_correct_variable_name_is_extracted(self, variable_codebook_fixture, valid_inputs):
        """
        Given: input contains keyword "<keyword>"
        When: variable is extracted from input
        Then: the variable_name is "<variable_name>"
        """
        # Arrange
        input_with_keyword = valid_inputs["sales_tax"]
        expected_var = "sales_tax_city"

        # Act
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_with_keyword)
        
        # Assert
        assert variable_name == expected_variable_name, f"Expected variable_name to be {expected_variable_name}, but got {variable_name}"

class TestCodebookOperationsAreLogged:
    """
    Tests for logging of codebook operations.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_run_is_called_then_log_message_indicates_operation_start(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given: any action is executed
        When: run is called
        Then: a log message indicates "Starting variable codebook operation: <action>"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert any('Starting variable codebook operation' in str(record.message) for record in caplog.records), f"Expected 'Starting variable codebook operation' in logs, but got {[str(r.message) for r in caplog.records]}"

    def test_when_variable_not_found_then_warning_is_logged(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given: variable_name "missing_var" does not exist
        When: get_variable is called
        Then: a warning is logged indicating "Variable not found: missing_var"
        """
        args, kwargs = valid_args["get_variable"], valid_kwargs["get_variable_missing"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert any('Variable not found: missing_var' in str(record.message) for record in caplog.records), f"Expected 'Variable not found: missing_var' in logs, but got {[str(r.message) for r in caplog.records]}"

    def test_when_decision_tree_is_missing_then_warning_is_logged(self, variable_codebook_fixture, valid_args, valid_kwargs):
        """
        Given a variable without prompt decision tree
        When get_prompt_sequence is called
        Then a warning is logged indicating "No prompt decision tree found"
        """
        args, kwargs = valid_args["get_prompt_sequence"], valid_kwargs["get_variable_test"]
        result = variable_codebook_fixture.run(*args, **kwargs)
        assert any('No prompt decision tree found' in str(record.message) for record in caplog.records), f"Expected 'No prompt decision tree found' in logs, but got {[str(r.message) for r in caplog.records]}"

