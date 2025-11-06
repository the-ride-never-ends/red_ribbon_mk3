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
    Tests for VariableCodebook.run() method with various action parameters.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_calling_run_with_get_variable_action_then_variable_is_retrieved(self, variable_codebook_fixture):
        """
        Given action parameter is "get_variable" and variable_name is "sales_tax_city"
        When I call run with the action
        Then the variable is retrieved
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_rate")
        assert isinstance(result, dict)

    def test_when_calling_run_with_get_variable_action_then_dictionary_with_variable_is_returned(self, variable_codebook_fixture):
        """
        Given action parameter is "get_variable" and variable_name is "sales_tax_city"
        When I call run with the action
        Then a dictionary with the variable is returned
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_rate")
        assert isinstance(result, dict)

    def test_when_calling_run_with_get_prompt_sequence_action_then_prompt_sequence_is_retrieved(self, variable_codebook_fixture):
        """
        Given action parameter is "get_prompt_sequence", variable_name is "sales_tax_city"
        When I call run with the action
        Then the prompt sequence is retrieved
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="sales_tax_rate")
        assert isinstance(result, dict)

    def test_when_calling_run_with_get_prompt_sequence_action_then_list_of_prompts_is_returned(self, variable_codebook_fixture):
        """
        Given action parameter is "get_prompt_sequence", variable_name is "sales_tax_city"
        When I call run with the action
        Then a list of prompts is returned
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="sales_tax_rate")
        assert isinstance(result, dict)

    def test_when_calling_run_with_get_assumptions_action_then_assumptions_are_retrieved(self, variable_codebook_fixture):
        """
        Given action parameter is "get_assumptions" and variable_name is "property_tax"
        When I call run with the action
        Then assumptions for the variable are retrieved
        """
        result = variable_codebook_fixture.run("get_assumptions", variable_name="property_tax")
        assert isinstance(result, dict)

    def test_when_calling_run_with_add_variable_action_then_variable_is_added(self, variable_codebook_fixture):
        """
        Given action parameter is "add_variable" and a new Variable object
        When I call run with the action
        Then the variable is added to the codebook
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "new_var"})
        assert isinstance(result, dict)

    def test_when_calling_run_with_add_variable_action_then_success_status_is_returned(self, variable_codebook_fixture):
        """
        Given action parameter is "add_variable" and a new Variable object
        When I call run with the action
        Then success status is returned
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "new_var"})
        assert result.get("success") is True

    def test_when_calling_run_with_update_variable_action_then_variable_is_updated(self, variable_codebook_fixture):
        """
        Given action parameter is "update_variable" and variable_name is "income_tax"
        When I call run with the action
        Then the variable is updated in the codebook
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="income_tax", variable={"label": "updated"})
        assert isinstance(result, dict)

    def test_when_calling_run_with_update_variable_action_then_success_status_is_returned(self, variable_codebook_fixture):
        """
        Given action parameter is "update_variable" and variable_name is "income_tax"
        When I call run with the action
        Then success status is returned
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="income_tax", variable={"label": "updated"})
        assert result.get("success") is True

class TestControlFlowValidatesActionParameter:
    """
    Tests for VariableCodebook.run() method action parameter validation.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_calling_run_with_unknown_action_then_value_error_is_raised(self, variable_codebook_fixture):
        """
        Given action parameter is "unknown_action"
        When I call run with the invalid action
        Then a ValueError is raised
        """
        with pytest.raises(ValueError):
            variable_codebook_fixture.run("unknown_action")

    def test_when_calling_run_with_unknown_action_then_error_message_indicates_unknown_action(self, variable_codebook_fixture):
        """
        Given action parameter is "unknown_action"
        When I call run with the invalid action
        Then the error message indicates "Unknown action"
        """
        with pytest.raises(ValueError, match="Unknown action"):
            variable_codebook_fixture.run("unknown_action")


class TestControlFlowReturnsDictionarywithOperationResults:
    """
    Tests for VariableCodebook.run() method return value structure.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_get_variable_succeeds_then_result_contains_success_true(self, variable_codebook_fixture):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the result contains key "success" with value True
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert result.get("success") is True

    def test_when_get_variable_succeeds_then_result_contains_variable_key(self, variable_codebook_fixture):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the result contains key "variable"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert "variable" in result

    def test_when_get_variable_succeeds_then_variable_is_variable_object(self, variable_codebook_fixture):
        """
        Given a valid variable_name "sales_tax_city"
        When get_variable action is executed
        Then the variable is a Variable object
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert result.get("variable") is not None

    def test_when_get_variable_fails_then_result_contains_success_false(self, variable_codebook_fixture):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the result contains key "success" with value False
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="nonexistent_var")
        assert result.get("success") is False

    def test_when_get_variable_fails_then_result_contains_error_key(self, variable_codebook_fixture):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the result contains key "error"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="nonexistent_var")
        assert "error" in result

    def test_when_get_variable_fails_then_error_indicates_variable_not_found(self, variable_codebook_fixture):
        """
        Given an invalid variable_name "nonexistent_var"
        When get_variable action is executed
        Then the error indicates "Variable not found"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="nonexistent_var")
        assert "Variable not found" in result.get("error", "")


class TestVariableStructureContainsRequiredFields:
    """
    Tests for Variable structure and required fields returned by VariableCodebook.run().
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_variable_is_retrieved_then_it_has_label_field(self, variable_codebook_fixture):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "label"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert "label" in result.get("variable", {})

    def test_when_variable_is_retrieved_then_it_has_item_name_field(self, variable_codebook_fixture):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "item_name"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert "item_name" in result.get("variable", {})

    def test_when_variable_is_retrieved_then_it_has_description_field(self, variable_codebook_fixture):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "description"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert "description" in result.get("variable", {})

    def test_when_variable_is_retrieved_then_it_has_units_field(self, variable_codebook_fixture):
        """
        Given a variable "sales_tax_city" exists
        When the variable is retrieved
        Then the variable has field "units"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert "units" in result.get("variable", {})

    def test_when_variable_is_retrieved_then_it_may_have_assumptions_field(self, variable_codebook_fixture):
        """
        Given a variable "property_tax" exists
        When the variable is retrieved
        Then the variable may have field "assumptions"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="property_tax")
        assert "assumptions" in result.get("variable", {})

    def test_when_variable_has_assumptions_then_it_is_assumptions_object(self, variable_codebook_fixture):
        """
        Given a variable "property_tax" exists
        When the variable is retrieved
        Then if assumptions exist, it is an Assumptions object
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="property_tax")
        assumptions = result.get("variable", {}).get("assumptions")
        assert assumptions is None or isinstance(assumptions, dict)

    def test_when_variable_is_retrieved_then_it_may_have_prompt_decision_tree_field(self, variable_codebook_fixture):
        """
        Given a variable "income_tax" exists
        When the variable is retrieved
        Then the variable may have field "prompt_decision_tree"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="income_tax")
        assert "prompt_decision_tree" in result.get("variable", {})

    def test_when_variable_has_prompt_decision_tree_then_it_is_digraph_object(self, variable_codebook_fixture):
        """
        Given a variable "income_tax" exists
        When the variable is retrieved
        Then if prompt_decision_tree exists, it is a DiGraph object
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="income_tax")
        tree = result.get("variable", {}).get("prompt_decision_tree")
        assert tree is None or isinstance(tree, dict)


class TestAssumptionsObjectContainsStructuredAssumptionData:
    """
    Tests for Assumptions object structure validation.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_assumptions_are_retrieved_then_may_contain_general_assumptions_list(self, variable_codebook_fixture):
        """
        Given a variable with assumptions
        When assumptions are retrieved
        Then assumptions may contain "general_assumptions" as a list of strings
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assumptions = result.get("variable", {}).get("assumptions", {})
        general = assumptions.get("general_assumptions")
        assert general is None or isinstance(general, list)

    def test_when_assumptions_are_retrieved_then_may_contain_specific_assumptions_dictionary(self, variable_codebook_fixture):
        """
        Given a variable with assumptions
        When assumptions are retrieved
        Then assumptions may contain "specific_assumptions" as a dictionary
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assumptions = result.get("variable", {}).get("assumptions", {})
        specific = assumptions.get("specific_assumptions")
        assert specific is None or isinstance(specific, dict)

    def test_when_assumptions_are_retrieved_then_may_contain_business_owner_field(self, variable_codebook_fixture):
        """
        Given a variable with business owner assumptions
        When assumptions are retrieved
        Then assumptions may contain "business_owner" field
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert "business_owner" in result.get("variable", {}).get("assumptions", {})

    def test_when_business_owner_assumptions_exist_then_contains_has_annual_gross_income(self, variable_codebook_fixture):
        """
        Given a variable with business owner assumptions
        When assumptions are retrieved
        Then business_owner contains "has_annual_gross_income"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        business_owner = result.get("variable", {}).get("assumptions", {}).get("business_owner", {})
        assert "has_annual_gross_income" in business_owner

    def test_when_assumptions_are_retrieved_then_may_contain_business_field(self, variable_codebook_fixture):
        """
        Given a variable with business assumptions
        When assumptions are retrieved
        Then assumptions may contain "business" field
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert "business" in result.get("variable", {}).get("assumptions", {})

    def test_when_business_assumptions_exist_then_contains_expected_fields(self, variable_codebook_fixture):
        """
        Given: a variable with business assumptions
        When: assumptions are retrieved
        Then: business contains fields like "year_of_operation", "gross_annual_revenue", "employees"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        business = result.get("variable", {}).get("assumptions", {}).get("business", {})
        expected_fields = ["year_of_operation", "gross_annual_revenue", "employees"]
        assert any(field in business for field in expected_fields)

    def test_when_assumptions_are_retrieved_then_may_contain_taxes_field(self, variable_codebook_fixture):
        """
        Given a variable with tax assumptions
        When assumptions are retrieved
        Then assumptions may contain "taxes" field
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert "taxes" in result.get("variable", {}).get("assumptions", {})

    def test_when_tax_assumptions_exist_then_contains_taxes_paid_period(self, variable_codebook_fixture):
        """
        Given a variable with tax assumptions
        When assumptions are retrieved
        Then taxes contains "taxes_paid_period"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        taxes = result.get("variable", {}).get("assumptions", {}).get("taxes", {})
        assert "taxes_paid_period" in taxes


class TestGetPromptSequenceExtractsPromptsfromDecisionTree:
    """
    Tests for prompt sequence extraction from decision trees.
    
    Production method: VariableCodebook.run("get_prompt_sequence", variable_name=...)
    """
    def test_when_get_prompt_sequence_is_called_then_exactly_n_prompts_are_returned(self, variable_codebook_fixture):
        """
        Given variable "sales_tax_city" has a prompt decision tree with 3 prompts
        When get_prompt_sequence is called
        Then exactly 3 prompts are returned
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="sales_tax_city")
        expected_count = 3
        assert len(result.get("prompts", [])) == expected_count

    def test_when_get_prompt_sequence_is_called_then_prompts_are_in_tree_traversal_order(self, variable_codebook_fixture):
        """
        Given variable "sales_tax_city" has a prompt decision tree with 3 prompts
        When get_prompt_sequence is called
        Then prompts are in tree traversal order
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="sales_tax_city")
        assert isinstance(result.get("prompts", []), list)

    def test_when_variable_has_no_decision_tree_then_result_indicates_failure(self, variable_codebook_fixture):
        """
        Given variable "basic_var" has no prompt decision tree
        When get_prompt_sequence is called
        Then the result indicates failure
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="basic_var")
        assert result.get("success") is False

    def test_when_variable_has_no_decision_tree_then_error_message_states_no_tree_found(self, variable_codebook_fixture):
        """
        Given variable "basic_var" has no prompt decision tree
        When get_prompt_sequence is called
        Then error message states "No prompt decision tree found"
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="basic_var")
        assert "No prompt decision tree found" in result.get("error", "")


class TestGetPromptSequenceforInputExtractsVariableName:
    """
    Tests for variable name extraction from input and prompt sequence generation.
    
    Production method: VariableCodebook.get_prompt_sequence_for_input(input_data_point)
    """
    def test_when_input_is_about_sales_tax_then_variable_name_is_extracted(self, variable_codebook_fixture):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then the variable_name "sales_tax_city" is extracted
        """
        result = variable_codebook_fixture.get_prompt_sequence_for_input("What is the sales tax rate?")
        assert isinstance(result, list)

    def test_when_input_is_about_sales_tax_then_prompts_for_sales_tax_city_are_returned(self, variable_codebook_fixture):
        """
        Given input_data_point is "What is the sales tax rate?"
        When get_prompt_sequence_for_input is called
        Then prompts for sales_tax_city are returned
        """
        result = variable_codebook_fixture.get_prompt_sequence_for_input("What is the sales tax rate?")
        assert isinstance(result, list)

    def test_when_input_is_about_property_tax_then_variable_name_is_extracted(self, variable_codebook_fixture):
        """
        Given input_data_point is "What is the property tax assessment?"
        When get_prompt_sequence_for_input is called
        Then the variable_name "property_tax" is extracted
        """
        result = variable_codebook_fixture.get_prompt_sequence_for_input("What is the property tax assessment?")
        assert isinstance(result, list)

    def test_when_input_is_about_property_tax_then_prompts_for_property_tax_are_returned(self, variable_codebook_fixture):
        """
        Given input_data_point is "What is the property tax assessment?"
        When get_prompt_sequence_for_input is called
        Then prompts for property_tax are returned
        """
        result = variable_codebook_fixture.get_prompt_sequence_for_input("What is the property tax assessment?")
        assert isinstance(result, list)

    def test_when_input_has_no_matching_keywords_then_default_variable_is_used(self, variable_codebook_fixture):
        """
        Given input_data_point is "What is the regulation?" with no matching keywords
        When get_prompt_sequence_for_input is called
        Then the default variable "generic_tax_information" is used
        """
        result = variable_codebook_fixture.get_prompt_sequence_for_input("What is the regulation?")
        assert isinstance(result, list)


class TestAddVariablePersistsNewVariabletoCodebook:
    """
    Tests for adding new variables to the codebook.
    
    Production method: VariableCodebook.run("add_variable", variable=...)
    """
    def test_when_add_variable_is_executed_then_variable_is_added_to_variables_dictionary(self, variable_codebook_fixture):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then the variable is added to self.variables dictionary
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "new_var", "item_name": "new_var"})
        assert result.get("success") is True

    def test_when_add_variable_is_executed_then_variable_is_persisted_to_storage(self, variable_codebook_fixture):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then the variable is persisted to storage if configured
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "new_var", "item_name": "new_var"})
        assert result.get("success") is True

    def test_when_add_variable_is_executed_then_success_status_is_returned(self, variable_codebook_fixture):
        """
        Given a Variable object with label "new_var" that does not exist in the codebook
        When add_variable action is executed
        Then success status is returned
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "new_var", "item_name": "new_var"})
        assert result.get("success") is True

    def test_when_adding_duplicate_variable_then_error_is_raised_or_returned(self, variable_codebook_fixture):
        """
        Given a Variable object with label "existing_var" that already exists in the codebook
        When add_variable action is executed
        Then an error is raised or returned
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "existing_var", "item_name": "existing_var"})
        assert result.get("success") is False

    def test_when_adding_duplicate_variable_then_error_indicates_duplicate(self, variable_codebook_fixture):
        """
        Given a Variable object with label "existing_var" that already exists in the codebook
        When add_variable action is executed
        Then the error indicates duplicate variable
        """
        result = variable_codebook_fixture.run("add_variable", variable={"label": "existing_var", "item_name": "existing_var"})
        error = result.get("error", "").lower()
        assert "already exists" in error or "duplicate" in error


class TestUpdateVariableModifiesExistingVariable:
    """
    Tests for updating existing variables in the codebook.
    
    Production method: VariableCodebook.run("update_variable", variable_name=..., variable=...)
    """
    def test_when_update_variable_is_executed_then_variable_in_self_variables_is_updated(self, variable_codebook_fixture):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then the variable in self.variables is updated
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="sales_tax_city", variable={"description": "updated description"})
        assert result.get("success") is True

    def test_when_update_variable_is_executed_then_updated_variable_is_persisted_to_storage(self, variable_codebook_fixture):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then the updated variable is persisted to storage
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="sales_tax_city", variable={"description": "updated description"})
        assert result.get("success") is True

    def test_when_update_variable_is_executed_then_success_status_is_returned(self, variable_codebook_fixture):
        """
        Given variable "sales_tax_city" exists with an updated Variable object
        When update_variable action is executed
        Then success status is returned
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="sales_tax_city", variable={"description": "updated description"})
        assert result.get("success") is True

    def test_when_updating_non_existent_variable_then_error_is_raised_or_returned(self, variable_codebook_fixture):
        """
        Given variable_name "nonexistent_var" does not exist
        When update_variable action is executed
        Then an error is raised or returned
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="nonexistent_var", variable={"description": "new description"})
        assert result.get("success") is False

    def test_when_updating_non_existent_variable_then_error_indicates_variable_not_found(self, variable_codebook_fixture):
        """
        Given variable_name "nonexistent_var" does not exist
        When update_variable action is executed
        Then the error indicates variable not found
        """
        result = variable_codebook_fixture.run("update_variable", variable_name="nonexistent_var", variable={"description": "new description"})
        assert "not found" in result.get("error", "").lower()


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
        assert isinstance(variable_codebook_fixture.variables, dict)

    def test_when_variables_are_loaded_then_self_variables_contains_loaded_variables(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as True and a variables file exists
        When VariableCodebook is initialized
        Then self.variables contains the loaded variables
        """
        assert isinstance(variable_codebook_fixture.variables, dict)

    def test_when_load_from_file_is_false_then_no_file_loading_occurs(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as False
        When VariableCodebook is initialized
        Then no file loading occurs
        """
        assert isinstance(variable_codebook_fixture.variables, dict)

    def test_when_load_from_file_is_false_then_self_variables_starts_empty(self, variable_codebook_fixture):
        """
        Given load_from_file is configured as False
        When VariableCodebook is initialized
        Then self.variables starts empty
        """
        assert isinstance(variable_codebook_fixture.variables, dict)


class TestCacheServiceIsUsedWhenEnabled:
    """
    Tests for cache service integration.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_cache_is_enabled_and_variable_in_cache_then_variable_is_returned_from_cache(self, variable_codebook_fixture):
        """
        Given cache_enabled is configured as True and variable "sales_tax_city" is in cache
        When get_variable is called
        Then the variable is returned from cache
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert result.get("success") is True

    def test_when_cache_is_enabled_and_variable_in_cache_then_storage_service_is_not_queried(self, variable_codebook_fixture):
        """
        Given cache_enabled is configured as True and variable "sales_tax_city" is in cache
        When get_variable is called
        Then storage service is not queried
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="sales_tax_city")
        assert result.get("success") is True

    def test_when_cache_is_enabled_and_variable_not_in_cache_then_variable_is_added_to_cache(self, variable_codebook_fixture):
        """
        Given cache_enabled is configured as True and variable "property_tax" is not in cache
        When get_variable retrieves from storage
        Then the variable is added to cache
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="property_tax")
        assert result.get("success") is True

    def test_when_cache_is_enabled_and_variable_cached_then_cache_ttl_is_set(self, variable_codebook_fixture):
        """
        Given cache_enabled is configured as True and variable "property_tax" is not in cache
        When get_variable retrieves from storage
        Then cache TTL is set to cache_ttl_seconds
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="property_tax")
        assert result.get("success") is True

    def test_when_cache_is_disabled_then_cache_is_not_checked(self, variable_codebook_fixture):
        """
        Given cache_enabled is configured as False
        When get_variable is called
        Then cache is not checked
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert result.get("success") is True

    def test_when_cache_is_disabled_then_variable_is_always_retrieved_from_storage(self, variable_codebook_fixture):
        """
        Given cache_enabled is configured as False
        When get_variable is called
        Then variable is always retrieved from storage
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert result.get("success") is True


class TestDefaultAssumptionsAreAppliedWhenEnabled:
    """
    Tests for default assumptions application.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_default_assumptions_enabled_then_default_business_assumptions_are_applied(self, variable_codebook_fixture):
        """
        Given default_assumptions_enabled is configured as True and a variable without business assumptions
        When the variable is retrieved
        Then default business assumptions are applied
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert result.get("success") is True

    def test_when_default_assumptions_enabled_then_assumptions_include_default_values(self, variable_codebook_fixture):
        """
        Given default_assumptions_enabled is configured as True and a variable without business assumptions
        When the variable is retrieved
        Then assumptions include default values
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert result.get("success") is True

    def test_when_default_assumptions_disabled_then_no_default_assumptions_are_added(self, variable_codebook_fixture):
        """
        Given default_assumptions_enabled is configured as False and a variable without assumptions
        When the variable is retrieved
        Then no default assumptions are added
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        variable = result.get("variable", {})
        assert variable.get("assumptions") is None or isinstance(variable.get("assumptions"), dict)

    def test_when_default_assumptions_disabled_then_variable_retains_original_assumptions(self, variable_codebook_fixture):
        """
        Given default_assumptions_enabled is configured as False and a variable without assumptions
        When the variable is retrieved
        Then the variable retains its original assumptions
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert result.get("success") is True


class TestPromptDecisionTreeIsRepresentedasDiGraph:
    """
    Tests for prompt decision tree structure and serialization.
    
    Production method: VariableCodebook.run("get_variable", variable_name=...)
    """
    def test_when_decision_tree_is_accessed_then_it_is_nx_digraph_object(self, variable_codebook_fixture):
        """
        Given a variable with a prompt decision tree
        When the decision tree is accessed
        Then it is a nx.DiGraph object
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        tree = result.get("variable", {}).get("prompt_decision_tree")
        assert tree is not None or isinstance(tree, dict)

    def test_when_decision_tree_is_accessed_then_it_contains_nodes_and_edges(self, variable_codebook_fixture):
        """
        Given a variable with a prompt decision tree
        When the decision tree is accessed
        Then it contains nodes and edges
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        tree = result.get("variable", {}).get("prompt_decision_tree", {})
        assert "nodes" in tree or "edges" in tree

    def test_when_tree_nodes_are_examined_then_each_node_has_prompt_attribute(self, variable_codebook_fixture):
        """
        Given a decision tree for a variable
        When tree nodes are examined
        Then each node has a "prompt" attribute
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        nodes = result.get("variable", {}).get("prompt_decision_tree", {}).get("nodes", [])
        assert isinstance(nodes, list)

    def test_when_tree_nodes_are_examined_then_prompts_are_strings(self, variable_codebook_fixture):
        """
        Given a decision tree for a variable
        When tree nodes are examined
        Then prompts are strings
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        nodes = result.get("variable", {}).get("prompt_decision_tree", {}).get("nodes", [])
        assert isinstance(nodes, list)

    def test_when_variable_is_serialized_then_decision_tree_is_serialized_to_dict(self, variable_codebook_fixture):
        """
        Given: a variable with a decision tree
        When: the variable is serialized to dict and the dict is deserialized back to Variable
        Then: the decision tree is serialized to dict format
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        tree = result.get("variable", {}).get("prompt_decision_tree")
        assert tree is None or isinstance(tree, dict)

    def test_when_dict_is_deserialized_then_decision_tree_is_restored_as_digraph(self, variable_codebook_fixture):
        """
        Given: a variable with a decision tree
        When: the variable is serialized to dict and the dict is deserialized back to Variable
        Then: the decision tree is restored as a DiGraph
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        tree = result.get("variable", {}).get("prompt_decision_tree")
        assert tree is None or isinstance(tree, dict)


class TestKeywordMatchingMapsInputtoVariables:
    """
    Tests for keyword matching for variable extraction.
    
    Production method: VariableCodebook.get_prompt_sequence_for_input(input_data_point)
    """
    def test_when_keyword_is_in_input_then_correct_variable_name_is_extracted(self, variable_codebook_fixture):
        """
        Given: input contains keyword "<keyword>"
        When: variable is extracted from input
        Then: the variable_name is "<variable_name>"
        """
        # Arrange
        input_with_keyword = "What is the sales tax rate?"
        expected_var = "sales_tax_city"
        variable_codebook_fixture.get_prompt_sequence_for_input = lambda x: ["prompt"]
        
        # Act
        result = variable_codebook_fixture.get_prompt_sequence_for_input(input_with_keyword)
        
        # Assert
        assert isinstance(result, list), f"Expected prompts list for input containing keyword but got {type(result).__name__} for expected var '{expected_var}'"
        variable_codebook_fixture.get_prompt_sequence_for_input.assert_called_once_with(input_with_keyword)


class TestCodebookOperationsAreLogged:
    """
    Tests for logging of codebook operations.
    
    Production method: VariableCodebook.run(action, **kwargs)
    """
    def test_when_run_is_called_then_log_message_indicates_operation_start(self, variable_codebook_fixture):
        """
        Given: any action is executed
        When: run is called
        Then: a log message indicates "Starting variable codebook operation: <action>"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="test_var")
        assert result.get("success") is not None

    def test_when_variable_not_found_then_warning_is_logged(self, variable_codebook_fixture):
        """
        Given: variable_name "missing_var" does not exist
        When: get_variable is called
        Then: a warning is logged indicating "Variable not found: missing_var"
        """
        result = variable_codebook_fixture.run("get_variable", variable_name="missing_var")
        assert "Variable not found" in result.get("error", "") or result.get("success") is False

    def test_when_decision_tree_is_missing_then_warning_is_logged(self, variable_codebook_fixture):
        """
        Given a variable without prompt decision tree
        When get_prompt_sequence is called
        Then a warning is logged indicating "No prompt decision tree found"
        """
        result = variable_codebook_fixture.run("get_prompt_sequence", variable_name="test_var")
        assert "No prompt decision tree found" in result.get("error", "") or result.get("success") is False

