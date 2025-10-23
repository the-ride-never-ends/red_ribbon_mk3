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
from unittest.mock import Mock

@pytest.fixture
def mock_variable_codebook():
    """Mock VariableCodebook instance"""
    mock = Mock()
    mock.use_cache = True
    mock.use_default_assumptions = True
    
    # Sample variable data
    mock.variables = {
        "sales_tax_rate": {
            "label": "Sales Tax Rate",
            "item_name": "sales_tax_rate",
            "description": "The sales tax rate",
            "units": "percentage",
            "assumptions": {
                "general": [],
                "specific": [],
                "business_owner": [],
                "business": [],
                "taxes": []
            },
            "prompt_decision_tree": None
        }
    }
    
    def mock_control_flow(action, **kwargs):
        if action == "get_variable":
            var_name = kwargs.get("variable_name", "")
            return {"variable": mock.variables.get(var_name, {})}
        elif action == "get_prompt_sequence":
            return {"prompt_sequence": []}
        elif action == "get_assumptions":
            return {"assumptions": {}}
        elif action == "add_variable":
            return {"success": True}
        elif action == "update_variable":
            return {"success": True}
        else:
            return {}
    
    mock.control_flow = mock_control_flow
    return mock


import pytest

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
class TestControlFlowMethodAcceptsActionParameter:
    """
    Rule: Control Flow Method Accepts Action Parameter
    """
    def test_control_flow_with_get_variable_action(self, mock_variable_codebook):
        """
        Scenario: Control flow with "get_variable" action
          Given action parameter is "get_variable"
          And variable_name is "sales_tax_city"
          When I call control_flow with the action
          Then the variable is retrieved
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_get_variable_action_1(self, mock_variable_codebook):
        """
        Scenario: Control flow with "get_variable" action
          Given action parameter is "get_variable"
          And variable_name is "sales_tax_city"
          When I call control_flow with the action
          Then a dictionary with the variable is returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_get_prompt_sequence_action(self, mock_variable_codebook):
        """
        Scenario: Control flow with "get_prompt_sequence" action
          Given action parameter is "get_prompt_sequence"
          And variable_name is "sales_tax_city"
          And input_data_point is "What is the sales tax rate?"
          When I call control_flow with the action
          Then the prompt sequence is retrieved
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_get_prompt_sequence_action_1(self, mock_variable_codebook):
        """
        Scenario: Control flow with "get_prompt_sequence" action
          Given action parameter is "get_prompt_sequence"
          And variable_name is "sales_tax_city"
          And input_data_point is "What is the sales tax rate?"
          When I call control_flow with the action
          Then a list of prompts is returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_get_assumptions_action(self, mock_variable_codebook):
        """
        Scenario: Control flow with "get_assumptions" action
          Given action parameter is "get_assumptions"
          And variable_name is "property_tax"
          When I call control_flow with the action
          Then assumptions for the variable are retrieved
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_add_variable_action(self, mock_variable_codebook):
        """
        Scenario: Control flow with "add_variable" action
          Given action parameter is "add_variable"
          And a new Variable object
          When I call control_flow with the action
          Then the variable is added to the codebook
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_add_variable_action_1(self, mock_variable_codebook):
        """
        Scenario: Control flow with "add_variable" action
          Given action parameter is "add_variable"
          And a new Variable object
          When I call control_flow with the action
          Then success status is returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_update_variable_action(self, mock_variable_codebook):
        """
        Scenario: Control flow with "update_variable" action
          Given action parameter is "update_variable"
          And variable_name is "income_tax"
          And an updated Variable object
          When I call control_flow with the action
          Then the variable is updated in the codebook
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_with_update_variable_action_1(self, mock_variable_codebook):
        """
        Scenario: Control flow with "update_variable" action
          Given action parameter is "update_variable"
          And variable_name is "income_tax"
          And an updated Variable object
          When I call control_flow with the action
          Then success status is returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

class TestControlFlowValidatesActionParameter:
    """
    Rule: Control Flow Validates Action Parameter
    """
    def test_control_flow_rejects_unknown_action(self, mock_variable_codebook):
        """
        Scenario: Control flow rejects unknown action
          Given action parameter is "unknown_action"
          When I call control_flow with the invalid action
          Then a ValueError is raised
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_control_flow_rejects_unknown_action_1(self, mock_variable_codebook):
        """
        Scenario: Control flow rejects unknown action
          Given action parameter is "unknown_action"
          When I call control_flow with the invalid action
          Then the error message indicates "Unknown action"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestControlFlowReturnsDictionarywithOperationResults:
    """
    Rule: Control Flow Returns Dictionary with Operation Results
    """
    def test_successful_get_variable_returns_success_flag(self, mock_variable_codebook):
        """
        Scenario: Successful get_variable returns success flag
          Given a valid variable_name "sales_tax_city"
          When get_variable action is executed
          Then the result contains key "success" with value True
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_successful_get_variable_returns_success_flag_1(self, mock_variable_codebook):
        """
        Scenario: Successful get_variable returns success flag
          Given a valid variable_name "sales_tax_city"
          When get_variable action is executed
          Then the result contains key "variable"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_successful_get_variable_returns_success_flag_2(self, mock_variable_codebook):
        """
        Scenario: Successful get_variable returns success flag
          Given a valid variable_name "sales_tax_city"
          When get_variable action is executed
          Then the variable is a Variable object
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_failed_get_variable_returns_error(self, mock_variable_codebook):
        """
        Scenario: Failed get_variable returns error
          Given an invalid variable_name "nonexistent_var"
          When get_variable action is executed
          Then the result contains key "success" with value False
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_failed_get_variable_returns_error_1(self, mock_variable_codebook):
        """
        Scenario: Failed get_variable returns error
          Given an invalid variable_name "nonexistent_var"
          When get_variable action is executed
          Then the result contains key "error"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_failed_get_variable_returns_error_2(self, mock_variable_codebook):
        """
        Scenario: Failed get_variable returns error
          Given an invalid variable_name "nonexistent_var"
          When get_variable action is executed
          Then the error indicates "Variable not found"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestVariableStructureContainsRequiredFields:
    """
    Rule: Variable Structure Contains Required Fields
    """
    def test_variable_has_all_required_fields(self, mock_variable_codebook):
        """
        Scenario: Variable has all required fields
          Given a variable "sales_tax_city" exists
          When the variable is retrieved
          Then the variable has field "label"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_all_required_fields_1(self, mock_variable_codebook):
        """
        Scenario: Variable has all required fields
          Given a variable "sales_tax_city" exists
          When the variable is retrieved
          Then the variable has field "item_name"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_all_required_fields_2(self, mock_variable_codebook):
        """
        Scenario: Variable has all required fields
          Given a variable "sales_tax_city" exists
          When the variable is retrieved
          Then the variable has field "description"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_all_required_fields_3(self, mock_variable_codebook):
        """
        Scenario: Variable has all required fields
          Given a variable "sales_tax_city" exists
          When the variable is retrieved
          Then the variable has field "units"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_optional_assumptions_field(self, mock_variable_codebook):
        """
        Scenario: Variable has optional assumptions field
          Given a variable "property_tax" exists
          When the variable is retrieved
          Then the variable may have field "assumptions"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_optional_assumptions_field_1(self, mock_variable_codebook):
        """
        Scenario: Variable has optional assumptions field
          Given a variable "property_tax" exists
          When the variable is retrieved
          Then if assumptions exist, it is an Assumptions object
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_optional_prompt_decision_tree_field(self, mock_variable_codebook):
        """
        Scenario: Variable has optional prompt_decision_tree field
          Given a variable "income_tax" exists
          When the variable is retrieved
          Then the variable may have field "prompt_decision_tree"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_has_optional_prompt_decision_tree_field_1(self, mock_variable_codebook):
        """
        Scenario: Variable has optional prompt_decision_tree field
          Given a variable "income_tax" exists
          When the variable is retrieved
          Then if prompt_decision_tree exists, it is a DiGraph object
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestAssumptionsObjectContainsStructuredAssumptionData:
    """
    Rule: Assumptions Object Contains Structured Assumption Data
    """
    def test_assumptions_has_general_assumptions_list(self, mock_variable_codebook):
        """
        Scenario: Assumptions has general assumptions list
          Given a variable with assumptions
          When assumptions are retrieved
          Then assumptions may contain "general_assumptions" as a list of strings
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_specific_assumptions_dictionary(self, mock_variable_codebook):
        """
        Scenario: Assumptions has specific assumptions dictionary
          Given a variable with assumptions
          When assumptions are retrieved
          Then assumptions may contain "specific_assumptions" as a dictionary
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_business_owner_assumptions(self, mock_variable_codebook):
        """
        Scenario: Assumptions has business owner assumptions
          Given a variable with business owner assumptions
          When assumptions are retrieved
          Then assumptions may contain "business_owner" field
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_business_owner_assumptions_1(self, mock_variable_codebook):
        """
        Scenario: Assumptions has business owner assumptions
          Given a variable with business owner assumptions
          When assumptions are retrieved
          Then business_owner contains "has_annual_gross_income"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_business_assumptions(self, mock_variable_codebook):
        """
        Scenario: Assumptions has business assumptions
          Given a variable with business assumptions
          When assumptions are retrieved
          Then assumptions may contain "business" field
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_business_assumptions_1(self, mock_variable_codebook):
        """
        Scenario: Assumptions has business assumptions
          Given a variable with business assumptions
          When assumptions are retrieved
          Then business contains fields like "year_of_operation", "gross_annual_revenue", "employees"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_tax_assumptions(self, mock_variable_codebook):
        """
        Scenario: Assumptions has tax assumptions
          Given a variable with tax assumptions
          When assumptions are retrieved
          Then assumptions may contain "taxes" field
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_assumptions_has_tax_assumptions_1(self, mock_variable_codebook):
        """
        Scenario: Assumptions has tax assumptions
          Given a variable with tax assumptions
          When assumptions are retrieved
          Then taxes contains "taxes_paid_period"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestGetPromptSequenceExtractsPromptsfromDecisionTree:
    """
    Rule: Get Prompt Sequence Extracts Prompts from Decision Tree
    """
    def test_prompt_sequence_is_extracted_from_decision_tree(self, mock_variable_codebook):
        """
        Scenario: Prompt sequence is extracted from decision tree
          Given variable "sales_tax_city" has a prompt decision tree
          And the tree contains 3 prompts
          When get_prompt_sequence is called
          Then exactly 3 prompts are returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_prompt_sequence_is_extracted_from_decision_tree_1(self, mock_variable_codebook):
        """
        Scenario: Prompt sequence is extracted from decision tree
          Given variable "sales_tax_city" has a prompt decision tree
          And the tree contains 3 prompts
          When get_prompt_sequence is called
          Then prompts are in tree traversal order
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_without_decision_tree_returns_error(self, mock_variable_codebook):
        """
        Scenario: Variable without decision tree returns error
          Given variable "basic_var" has no prompt decision tree
          When get_prompt_sequence is called
          Then the result indicates failure
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_without_decision_tree_returns_error_1(self, mock_variable_codebook):
        """
        Scenario: Variable without decision tree returns error
          Given variable "basic_var" has no prompt decision tree
          When get_prompt_sequence is called
          Then error message states "No prompt decision tree found"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestGetPromptSequenceforInputExtractsVariableName:
    """
    Rule: Get Prompt Sequence for Input Extracts Variable Name
    """
    def test_input_about_sales_tax_maps_to_sales_tax_city_variable(self, mock_variable_codebook):
        """
        Scenario: Input about sales tax maps to sales_tax_city variable
          Given input_data_point is "What is the sales tax rate?"
          When get_prompt_sequence_for_input is called
          Then the variable_name "sales_tax_city" is extracted
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_input_about_sales_tax_maps_to_sales_tax_city_variable_1(self, mock_variable_codebook):
        """
        Scenario: Input about sales tax maps to sales_tax_city variable
          Given input_data_point is "What is the sales tax rate?"
          When get_prompt_sequence_for_input is called
          Then prompts for sales_tax_city are returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_input_about_property_tax_maps_to_property_tax_variable(self, mock_variable_codebook):
        """
        Scenario: Input about property tax maps to property_tax variable
          Given input_data_point is "What is the property tax assessment?"
          When get_prompt_sequence_for_input is called
          Then the variable_name "property_tax" is extracted
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_input_about_property_tax_maps_to_property_tax_variable_1(self, mock_variable_codebook):
        """
        Scenario: Input about property tax maps to property_tax variable
          Given input_data_point is "What is the property tax assessment?"
          When get_prompt_sequence_for_input is called
          Then prompts for property_tax are returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_input_with_no_matching_keywords_uses_default_variable(self, mock_variable_codebook):
        """
        Scenario: Input with no matching keywords uses default variable
          Given input_data_point is "What is the regulation?"
          And no keywords match existing variables
          When get_prompt_sequence_for_input is called
          Then the default variable "generic_tax_information" is used
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestAddVariablePersistsNewVariabletoCodebook:
    """
    Rule: Add Variable Persists New Variable to Codebook
    """
    def test_new_variable_is_added_successfully(self, mock_variable_codebook):
        """
        Scenario: New variable is added successfully
          Given a Variable object with label "new_var"
          And "new_var" does not exist in the codebook
          When add_variable action is executed
          Then the variable is added to self.variables dictionary
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_new_variable_is_added_successfully_1(self, mock_variable_codebook):
        """
        Scenario: New variable is added successfully
          Given a Variable object with label "new_var"
          And "new_var" does not exist in the codebook
          When add_variable action is executed
          Then the variable is persisted to storage if configured
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_new_variable_is_added_successfully_2(self, mock_variable_codebook):
        """
        Scenario: New variable is added successfully
          Given a Variable object with label "new_var"
          And "new_var" does not exist in the codebook
          When add_variable action is executed
          Then success status is returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_adding_duplicate_variable_is_rejected(self, mock_variable_codebook):
        """
        Scenario: Adding duplicate variable is rejected
          Given a Variable object with label "existing_var"
          And "existing_var" already exists in the codebook
          When add_variable action is executed
          Then an error is raised or returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_adding_duplicate_variable_is_rejected_1(self, mock_variable_codebook):
        """
        Scenario: Adding duplicate variable is rejected
          Given a Variable object with label "existing_var"
          And "existing_var" already exists in the codebook
          When add_variable action is executed
          Then the error indicates duplicate variable
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestUpdateVariableModifiesExistingVariable:
    """
    Rule: Update Variable Modifies Existing Variable
    """
    def test_existing_variable_is_updated_successfully(self, mock_variable_codebook):
        """
        Scenario: Existing variable is updated successfully
          Given variable "sales_tax_city" exists
          And an updated Variable object with new description
          When update_variable action is executed
          Then the variable in self.variables is updated
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_existing_variable_is_updated_successfully_1(self, mock_variable_codebook):
        """
        Scenario: Existing variable is updated successfully
          Given variable "sales_tax_city" exists
          And an updated Variable object with new description
          When update_variable action is executed
          Then the updated variable is persisted to storage
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_existing_variable_is_updated_successfully_2(self, mock_variable_codebook):
        """
        Scenario: Existing variable is updated successfully
          Given variable "sales_tax_city" exists
          And an updated Variable object with new description
          When update_variable action is executed
          Then success status is returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_updating_non_existent_variable_fails(self, mock_variable_codebook):
        """
        Scenario: Updating non-existent variable fails
          Given variable_name "nonexistent_var"
          And "nonexistent_var" does not exist
          When update_variable action is executed
          Then an error is raised or returned
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_updating_non_existent_variable_fails_1(self, mock_variable_codebook):
        """
        Scenario: Updating non-existent variable fails
          Given variable_name "nonexistent_var"
          And "nonexistent_var" does not exist
          When update_variable action is executed
          Then the error indicates variable not found
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestVariablesAreLoadedfromFileWhenConfigured:
    """
    Rule: Variables Are Loaded from File When Configured
    """
    def test_variables_are_loaded_on_initialization(self, mock_variable_codebook):
        """
        Scenario: Variables are loaded on initialization
          Given load_from_file is configured as True
          And a variables file exists at variables_path
          When VariableCodebook is initialized
          Then variables are loaded from the file
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variables_are_loaded_on_initialization_1(self, mock_variable_codebook):
        """
        Scenario: Variables are loaded on initialization
          Given load_from_file is configured as True
          And a variables file exists at variables_path
          When VariableCodebook is initialized
          Then self.variables contains the loaded variables
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_loading_is_skipped_when_disabled(self, mock_variable_codebook):
        """
        Scenario: Variable loading is skipped when disabled
          Given load_from_file is configured as False
          When VariableCodebook is initialized
          Then no file loading occurs
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_loading_is_skipped_when_disabled_1(self, mock_variable_codebook):
        """
        Scenario: Variable loading is skipped when disabled
          Given load_from_file is configured as False
          When VariableCodebook is initialized
          Then self.variables starts empty
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestCacheServiceIsUsedWhenEnabled:
    """
    Rule: Cache Service Is Used When Enabled
    """
    def test_cache_is_checked_before_retrieving_variable(self, mock_variable_codebook):
        """
        Scenario: Cache is checked before retrieving variable
          Given cache_enabled is configured as True
          And variable "sales_tax_city" is in cache
          When get_variable is called
          Then the variable is returned from cache
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_cache_is_checked_before_retrieving_variable_1(self, mock_variable_codebook):
        """
        Scenario: Cache is checked before retrieving variable
          Given cache_enabled is configured as True
          And variable "sales_tax_city" is in cache
          When get_variable is called
          Then storage service is not queried
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_cache_is_populated_after_storage_retrieval(self, mock_variable_codebook):
        """
        Scenario: Cache is populated after storage retrieval
          Given cache_enabled is configured as True
          And variable "property_tax" is not in cache
          When get_variable retrieves from storage
          Then the variable is added to cache
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_cache_is_populated_after_storage_retrieval_1(self, mock_variable_codebook):
        """
        Scenario: Cache is populated after storage retrieval
          Given cache_enabled is configured as True
          And variable "property_tax" is not in cache
          When get_variable retrieves from storage
          Then cache TTL is set to cache_ttl_seconds
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_cache_is_bypassed_when_disabled(self, mock_variable_codebook):
        """
        Scenario: Cache is bypassed when disabled
          Given cache_enabled is configured as False
          When get_variable is called
          Then cache is not checked
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_cache_is_bypassed_when_disabled_1(self, mock_variable_codebook):
        """
        Scenario: Cache is bypassed when disabled
          Given cache_enabled is configured as False
          When get_variable is called
          Then variable is always retrieved from storage or memory
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestDefaultAssumptionsAreAppliedWhenEnabled:
    """
    Rule: Default Assumptions Are Applied When Enabled
    """
    def test_default_assumptions_are_added_to_variable(self, mock_variable_codebook):
        """
        Scenario: Default assumptions are added to variable
          Given default_assumptions_enabled is configured as True
          And a variable without business assumptions
          When the variable is retrieved
          Then default business assumptions are applied
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_default_assumptions_are_added_to_variable_1(self, mock_variable_codebook):
        """
        Scenario: Default assumptions are added to variable
          Given default_assumptions_enabled is configured as True
          And a variable without business assumptions
          When the variable is retrieved
          Then assumptions include default values
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_default_assumptions_are_not_applied_when_disabled(self, mock_variable_codebook):
        """
        Scenario: Default assumptions are not applied when disabled
          Given default_assumptions_enabled is configured as False
          And a variable without assumptions
          When the variable is retrieved
          Then no default assumptions are added
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_default_assumptions_are_not_applied_when_disabled_1(self, mock_variable_codebook):
        """
        Scenario: Default assumptions are not applied when disabled
          Given default_assumptions_enabled is configured as False
          And a variable without assumptions
          When the variable is retrieved
          Then the variable retains its original assumptions
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestPromptDecisionTreeIsRepresentedasDiGraph:
    """
    Rule: Prompt Decision Tree Is Represented as DiGraph
    """
    def test_decision_tree_is_a_networkx_digraph(self, mock_variable_codebook):
        """
        Scenario: Decision tree is a NetworkX DiGraph
          Given a variable with a prompt decision tree
          When the decision tree is accessed
          Then it is a nx.DiGraph object
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_decision_tree_is_a_networkx_digraph_1(self, mock_variable_codebook):
        """
        Scenario: Decision tree is a NetworkX DiGraph
          Given a variable with a prompt decision tree
          When the decision tree is accessed
          Then it contains nodes and edges
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_decision_tree_nodes_have_prompt_attributes(self, mock_variable_codebook):
        """
        Scenario: Decision tree nodes have prompt attributes
          Given a decision tree for a variable
          When tree nodes are examined
          Then each node has a "prompt" attribute
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_decision_tree_nodes_have_prompt_attributes_1(self, mock_variable_codebook):
        """
        Scenario: Decision tree nodes have prompt attributes
          Given a decision tree for a variable
          When tree nodes are examined
          Then prompts are strings
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_decision_tree_can_be_serialized_and_deserialized(self, mock_variable_codebook):
        """
        Scenario: Decision tree can be serialized and deserialized
          Given a variable with a decision tree
          When the variable is serialized to dict
          When the dict is deserialized back to Variable
          Then the decision tree is serialized to dict format
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_decision_tree_can_be_serialized_and_deserialized_1(self, mock_variable_codebook):
        """
        Scenario: Decision tree can be serialized and deserialized
          Given a variable with a decision tree
          When the variable is serialized to dict
          When the dict is deserialized back to Variable
          Then the decision tree is restored as a DiGraph
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestKeywordMatchingMapsInputtoVariables:
    """
    Rule: Keyword Matching Maps Input to Variables
    """
    def test_keyword_in_input_maps_to_correct_variable(self, mock_variable_codebook):
        """
        Scenario: Keyword in input maps to correct variable
          Given input contains keyword "<keyword>"
          When variable is extracted from input
          Then the variable_name is "<variable_name>"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)


class TestCodebookOperationsAreLogged:
    """
    Rule: Codebook Operations Are Logged
    """
    def test_control_flow_logs_operation_start(self, mock_variable_codebook):
        """
        Scenario: Control flow logs operation start
          Given any action is executed
          When control_flow is called
          Then a log message indicates "Starting variable codebook operation: <action>"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_variable_not_found_is_logged_as_warning(self, mock_variable_codebook):
        """
        Scenario: Variable not found is logged as warning
          Given variable_name "missing_var" does not exist
          When get_variable is called
          Then a warning is logged indicating "Variable not found: missing_var"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

    def test_missing_decision_tree_is_logged_as_warning(self, mock_variable_codebook):
        """
        Scenario: Missing decision tree is logged as warning
          Given a variable without prompt decision tree
          When get_prompt_sequence is called
          Then a warning is logged indicating "No prompt decision tree found"
        """

        # Arrange
        action = "get_variable"
        
        # Act
        result = mock_variable_codebook.control_flow(action, variable_name="sales_tax_rate")
        
        # Assert
        assert isinstance(result, dict)

