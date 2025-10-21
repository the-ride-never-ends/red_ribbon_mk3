Feature: Prompt Decision Tree
  As a user 
  I want to extract ordinance information from input documents

  Background:
    Given an initialized PromptDecisionTree object.
    And an LLM api client is available
    And a database connection is available

  Rule: `extract_ordinance_information` Always Returns Dictionary
    Background:
      Given a valid string containing ordinance information

    Scenario: Extract ordinance information as a dictionary
      Given a valid string that contains ordinance information
      When I run `extract_ordinance_information` on the string
      Then I should receive a dictionary.

  Rule: Dictionary Always Contains Expected Keys

  Rule: Dictionary Keys Always Have Expected Value Types

  Rule: Dictionary Keys Always Have Expected Value Formats

  Rule: Tree Always Rejects Invalid Input Types

  Rule: Tree Always Rejects Invalid Input Values

  Rule: Tree 

  Scenario: Request a non-existent file
    Given the filename does not exist
    When I request the file
    Then I should receive an HTTPException

  Scenario: Request a path that is not a file
    Given the filename points to a directory or invalid path
    When I request the file
    Then I should receive an HTTPException

  Scenario: Request an unsupported file type
    Given the filename has an unsupported file extension
    When I request the file
    Then I should receive an HTTPException



