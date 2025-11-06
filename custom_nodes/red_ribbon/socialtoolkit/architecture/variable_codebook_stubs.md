# Function and Class stubs from '/home/kylerose1946/red_ribbon_mk3/custom_nodes/red_ribbon/socialtoolkit/architecture/variable_codebook.py'

Files last updated: 1762211658.3494794

Stub file last updated: 2025-11-03 15:16:24

## Assumptions

```python
class Assumptions(BaseModel):
    """
    Comprehensive model aggregating all assumption categories for variable analysis.

Attributes:
    general_assumptions (Optional[list[str]], optional): List of general
        assumptions applicable across all categories. Defaults to None.
    specific_assumptions (Optional[dict[str, str]], optional): Dictionary
        mapping specific assumption names to their values. Defaults to None.
    aspect (Optional[str], optional): Specific aspect or focus area for
        the assumptions. Defaults to None.
    business_owner (Optional[BusinessOwnerAssumptions], optional): Business
        owner-specific assumptions. Defaults to None.
    business (Optional[BusinessAssumptions], optional): Business operational
        assumptions. Defaults to None.
    taxes (Optional[TaxesAssumptions], optional): Tax-related assumptions.
        Defaults to None.
    other (Optional[OtherAssumptions], optional): Miscellaneous assumptions.
        Defaults to None.

Examples:
    >>> # Basic assumptions with categories
    >>> assumptions = Assumptions(
    ...     business_owner=BusinessOwnerAssumptions(),
    ...     business=BusinessAssumptions(),
    ...     aspect="sales_tax_analysis"
    ... )
    >>> 
    >>> # Comprehensive assumptions
    >>> full_assumptions = Assumptions(
    ...     general_assumptions=["Standard business practices apply"],
    ...     specific_assumptions={"tax_year": "2024"},
    ...     business_owner=BusinessOwnerAssumptions(),
    ...     business=BusinessAssumptions(),
    ...     taxes=TaxesAssumptions(),
    ...     other=OtherAssumptions(
    ...         other_assumptions=["No special exemptions"]
    ...     )
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## BusinessAssumptions

```python
class BusinessAssumptions(BaseModel):
    """
    Model representing assumptions about business operational characteristics.

Attributes:
    year_of_operation (str, optional): The operational stage of the business.
        Defaults to "second year".
    qualifies_for_incentives (bool, optional): Whether the business qualifies
        for tax incentives or special programs. Defaults to False.
    gross_annual_revenue (str, optional): The assumed annual revenue of the
        business. Defaults to "$1,000,000".
    employees (int, optional): The assumed number of employees. Defaults to 15.
    business_type (str, optional): The type and classification of business
        activities. Defaults to "general commercial activities (NAICS: 4523)".

Examples:
    >>> assumptions = BusinessAssumptions()
    >>> print(f"Revenue: {assumptions.gross_annual_revenue}")
    Revenue: $1,000,000
    >>> print(f"Employees: {assumptions.employees}")
    Employees: 15
    >>> 
    >>> custom_assumptions = BusinessAssumptions(
    ...     year_of_operation="first year",
    ...     qualifies_for_incentives=True,
    ...     employees=5
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## BusinessOwnerAssumptions

```python
class BusinessOwnerAssumptions(BaseModel):
    """
    Model representing assumptions about business ownership characteristics.

Attributes:
    annual_gross_income (str, optional): The assumed annual gross income
        of the business owner. Defaults to "$70,000".

Attributes:
    annual_gross_income (str): Business owner's assumed annual gross income
        formatted as a currency string.

Examples:
    >>> assumptions = BusinessOwnerAssumptions()
    >>> print(assumptions.annual_gross_income)
    '$70,000'
    >>> 
    >>> custom_assumptions = BusinessOwnerAssumptions(
    ...     annual_gross_income="$85,000"
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## CodeBook

```python
class CodeBook(BaseModel):
    """
    Model representing a collection of variable definitions for analysis workflows.

Attributes:
    variables (dict[str, Variable]): Dictionary mapping variable names
        to their corresponding Variable instances. Keys should match
        the item_name of each Variable.

Examples:
    >>> variables_dict = {
    ...     "sales_tax_city": Variable(
    ...         label="Sales Tax - City",
    ...         item_name="sales_tax_city",
    ...         description="Municipal sales tax rate",
    ...         units="Double (Percent)"
    ...     ),
    ...     "property_tax": Variable(
    ...         label="Property Tax",
    ...         item_name="property_tax",
    ...         description="Annual property tax rate",
    ...         units="Double (Percent)"
    ...     )
    ... }
    >>> codebook = CodeBook(variables=variables_dict)
    >>> print(len(codebook.variables))
    2
    >>> print("sales_tax_city" in codebook.variables)
    True
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## OtherAssumptions

```python
class OtherAssumptions(BaseModel):
    """
    Model representing miscellaneous assumptions not covered by other categories.

Attributes:
    other_assumptions (list[str], optional): List of additional assumptions
        as descriptive strings. Defaults to an empty list.

Examples:
    >>> assumptions = OtherAssumptions()
    >>> print(len(assumptions.other_assumptions))
    0
    >>> 
    >>> custom_assumptions = OtherAssumptions(
    ...     other_assumptions=[
    ...         "Business operates in multiple states",
    ...         "No special exemptions apply"
    ...     ]
    ... )
    >>> print(len(custom_assumptions.other_assumptions))
    2
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## PromptDecisionTree

```python
class PromptDecisionTree(BaseModel):
    """
    Model representing a complete prompt decision tree for variable extraction workflows.

This class combines nodes and edges to create a structured decision tree that
guides prompt sequencing and variable analysis. It integrates with NetworkX
to provide graph-based navigation and supports complex conditional logic
for sophisticated extraction workflows.

Args:
    nodes (list[PromptDecisionTreeNode], optional): List of nodes in the
        decision tree. Defaults to an empty list.
    edges (list[PromptDecisionTreeEdge], optional): List of edges connecting
        nodes in the decision tree. Defaults to an empty list.
    **data: Additional keyword arguments passed to NetworkX DiGraph initialization.

Attributes:
    nodes (list[PromptDecisionTreeNode]): Collection of decision tree nodes
    edges (list[PromptDecisionTreeEdge]): Collection of decision tree edges
    _tree (dict): Internal tree structure representation
    _graph (Optional[nx.DiGraph]): NetworkX graph representation of the tree

Key Features:
- Complete decision tree structure with nodes and edges
- NetworkX integration for graph-based operations
- Flexible tree construction and navigation
- Support for conditional edge traversal
- Structured prompt sequencing workflows

Raises:
    ValueError: If required 'nodes' or 'edges' fields are not provided,
        or if NetworkX DiGraph initialization fails.

Examples:
    >>> nodes = [
    ...     PromptDecisionTreeNode(name="start", prompt="Initial question"),
    ...     PromptDecisionTreeNode(name="follow_up", prompt="Follow-up question")
    ... ]
    >>> edges = [
    ...     PromptDecisionTreeEdge(prev_node="start", to_node="follow_up")
    ... ]
    >>> tree = PromptDecisionTree(nodes=nodes, edges=edges)
    >>> print(len(tree.nodes))
    2
    >>> print(len(tree.edges))
    1
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## PromptDecisionTreeEdge

```python
class PromptDecisionTreeEdge(BaseModel):
    """
    Model representing a directed edge between nodes in a prompt decision tree.

This class defines connections between decision tree nodes, establishing
the flow and conditional logic for prompt sequences. Edges can include
optional conditions that determine when transitions between nodes should
occur during the analysis workflow.

Args:
    prev_node (str): Identifier of the source node for this edge.
        Must correspond to an existing node name in the decision tree.
    to_node (str): Identifier of the target node for this edge.
        Must correspond to an existing node name in the decision tree.
    condition (Optional[Callable], optional): Optional function that determines
        whether this edge should be traversed. Defaults to None.

Attributes:
    prev_node (str): Source node identifier for edge traversal
    to_node (str): Target node identifier for edge traversal
    condition (Optional[Callable]): Optional conditional logic for edge traversal

Properties:
    args (tuple): Returns source and target nodes as tuple for graph construction
    kwargs (dict): Returns condition as keyword arguments for graph construction

Key Features:
- Directed edge definition between decision tree nodes
- Optional conditional logic for edge traversal
- Graph construction integration via properties
- Flexible workflow routing support

Examples:
    >>> edge = PromptDecisionTreeEdge(
    ...     prev_node="initial_check",
    ...     to_node="detailed_analysis",
    ...     condition=None
    ... )
    >>> print(edge.args)
    ('initial_check', 'detailed_analysis')
    >>> print(edge.kwargs)
    {'condition': None}
    >>> 
    >>> # Edge with condition
    >>> conditional_edge = PromptDecisionTreeEdge(
    ...     prev_node="check_result",
    ...     to_node="next_step",
    ...     condition=lambda x: x > 0.5
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## PromptDecisionTreeNode

```python
class PromptDecisionTreeNode(BaseModel):
    """
    Model representing a single node in a prompt decision tree structure.

This class defines individual nodes within a decision tree used for prompt
sequencing and variable extraction workflows. Each node contains a unique
identifier and associated prompt text that guides the analysis process
through structured questioning sequences.

Args:
    name (str): Unique identifier for the node within the decision tree.
        Must be unique across all nodes in the same tree structure.
    prompt (str): The prompt text or question associated with this node.
        Used to guide analysis or extraction at this decision point.

Attributes:
    name (str): Unique node identifier for tree navigation and referencing
    prompt (str): Associated prompt text for this decision point

Properties:
    args (tuple): Returns node name as a tuple for graph construction
    kwargs (dict): Returns prompt as keyword arguments for graph construction

Key Features:
- Unique node identification within decision trees
- Structured prompt text association
- Graph construction integration via properties
- Decision tree navigation support

Examples:
    >>> node = PromptDecisionTreeNode(
    ...     name="initial_check",
    ...     prompt="Does the document contain tax information?"
    ... )
    >>> print(node.name)
    initial_check
    >>> print(node.args)
    ('initial_check',)
    >>> print(node.kwargs)
    {'prompt': 'Does the document contain tax information?'}
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## TaxesAssumptions

```python
class TaxesAssumptions(BaseModel):
    """
    Model representing assumptions about tax-related business characteristics.

Attributes:
    taxes_paid_period (str, optional): The period during which taxes are
        assumed to be paid. Defaults to "second year of operation".

Examples:
    >>> assumptions = TaxesAssumptions()
    >>> print(assumptions.taxes_paid_period)
    second year of operation
    >>> 
    >>> custom_assumptions = TaxesAssumptions(
    ...     taxes_paid_period="third year of operation"
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## Variable

```python
class Variable(BaseModel):
    """
    Model representing a complete variable definition in the codebook system.

Attributes:
    label (str): Human-readable label for the variable. Used for display
        and user interface purposes.
    item_name (str): Internal identifier for the variable. Used for
        programmatic access and database operations.
    description (str): Detailed description explaining the variable's
        purpose and meaning in analysis contexts.
    units (str): Units of measurement or data type for the variable.
        Examples include "Double (Percent)", "Currency", "Boolean".
    assumptions (Optional[Assumptions], optional): Associated assumptions
        for this variable. Defaults to None.
    prompt_decision_tree (Optional[PromptDecisionTree], optional): Decision
        tree structure for extracting this variable. Defaults to None.

Examples:
    >>> variable = Variable(
    ...     label="Sales Tax - City",
    ...     item_name="sales_tax_city",
    ...     description="Municipal sales tax rate",
    ...     units="Double (Percent)"
    ... )
    >>> print(variable.label)
    Sales Tax - City
    >>> 
    >>> # Variable with assumptions and decision tree
    >>> complete_variable = Variable(
    ...     label="Property Tax Rate",
    ...     item_name="property_tax_rate",
    ...     description="Annual property tax assessment rate",
    ...     units="Double (Percent)",
    ...     assumptions=Assumptions(business=BusinessAssumptions()),
    ...     prompt_decision_tree=PromptDecisionTree(nodes=[], edges=[])
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## VariableCodebook

```python
class VariableCodebook:
    """
    Variable Codebook system for managing variable definitions and analysis workflows.

The VariableCodebook class provides comprehensive functionality for managing
variable definitions, their associated assumptions, and prompt sequences used
in tax and business analysis. It integrates with storage services, caching
systems, and language models to provide a complete variable management and
extraction workflow system.

Args:
    resources (dict[str, Callable]): Dictionary of injected dependencies including
        database, logger, storage services, cache services, and language models.
    configs (VariableCodebookConfigs): Configuration object controlling system
        behavior, file paths, caching, and default assumptions.

Key Features:
- Comprehensive variable definition management with CRUD operations
- Integrated assumption framework for business and tax analysis
- Prompt decision tree support for structured variable extraction
- Configurable storage and caching integration
- Language model integration for intelligent variable processing
- File-based and database-backed variable persistence
- Default variable and assumption loading capabilities

Attributes:
    resources (dict[str, Callable]): Injected service dependencies
    configs (VariableCodebookConfigs): System configuration settings
    variables (dict[str, Variable]): Collection of managed variables
    storage_service: Injected storage service for data persistence
    cache_service: Injected caching service for performance optimization
    logger (logging.Logger): Injected logging service for system monitoring

Public Methods:
    run(action: str, **kwargs) -> dict[str, Any]:
        Execute variable codebook operations including get, add, update operations
        and prompt sequence generation based on action type.
    get_prompt_sequence_for_input(input_data_point: str) -> list[str]:
        Generate prompt sequences for variable extraction from input queries.
        Analyzes input to determine relevant variable and returns structured prompts.

Usage Example:
>>> resources = {
...     'db': database_service,
...     'logger': logging.getLogger(),
...     'llm': language_model,
...     'storage_service': storage_service,
...     'cache_service': cache_service
... }
>>> configs = VariableCodebookConfigs(cache_enabled=True)
>>> codebook = VariableCodebook(resources, configs)
>>> 
>>> # Get a variable
>>> result = codebook.run("get_variable", variable_name="sales_tax_city")
>>> 
>>> # Generate prompt sequence
>>> prompts = codebook.get_prompt_sequence_for_input("What is the city sales tax rate?")
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## VariableCodebookConfigs

```python
class VariableCodebookConfigs(BaseModel):
    """
    Configuration model for Variable Codebook system initialization and behavior.

Attributes:
    variables_path (str, optional): Path to the variables data file for
        loading and saving operations. Defaults to "variables.json".
    load_from_file (bool, optional): Whether to load variables from file
        during initialization. Defaults to True.
    cache_enabled (bool, optional): Whether to enable caching for variable
        operations and lookups. Defaults to True.
    cache_ttl_seconds (int, optional): Cache time-to-live in seconds for
        cached variable data. Defaults to 3600 (1 hour).
    default_assumptions_enabled (bool, optional): Whether to load default
        assumptions when no file data is available. Defaults to True.

Examples:
    >>> # Default configuration
    >>> config = VariableCodebookConfigs()
    >>> print(config.variables_path)
    variables.json
    >>> print(config.cache_ttl_seconds)
    3600
    >>> 
    >>> # Custom configuration for development
    >>> dev_config = VariableCodebookConfigs(
    ...     variables_path="/dev/test_variables.json",
    ...     cache_enabled=False,
    ...     default_assumptions_enabled=False
    ... )
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## __init__

```python
def __init__(self, **data: Any):
    """
    Initialize a VariableCodebook instance with injected dependencies and configuration.

Args:
    resources (dict[str, Callable]): Dictionary of service dependencies with the following keys:
        - 'db': Database service for data persistence operations
        - 'logger': Logging service for system monitoring and debugging
        - 'llm': Language model service for intelligent text processing
        - 'storage_service' (optional): Storage service for variable file operations
        - 'cache_service' (optional): Caching service for performance optimization
    configs (VariableCodebookConfigs): Configuration object controlling:
        - File loading behavior and paths
        - Caching settings and time-to-live values
        - Default assumption loading preferences

Raises:
    TypeError: If resources is not a dictionary or configs is not a
        VariableCodebookConfigs instance.
    KeyError: If required service keys ('db', 'logger', 'llm') are missing
        from the resources dictionary.
    ValueError: If service instances are None or invalid.
    FileNotFoundError: If configured variables file path does not exist
        and load_from_file is True.
    RuntimeError: If service initialization or variable loading fails
        unexpectedly.

Examples:
    >>> resources = {
    ...     'db': database_service,
    ...     'logger': logging.getLogger('codebook'),
    ...     'llm': language_model_service,
    ...     'storage_service': file_storage_service,
    ...     'cache_service': redis_cache_service
    ... }
    >>> configs = VariableCodebookConfigs(
    ...     variables_path="production_variables.json",
    ...     cache_enabled=True
    ... )
    >>> codebook = VariableCodebook(resources, configs)
    """
```
* **Async:** False
* **Method:** True
* **Class:** PromptDecisionTree

## __init__

```python
def __init__(self, resources: dict[str, Callable], configs: VariableCodebookConfigs):
    """
    Initialize with injected dependencies and configuration

Args:
    resources: Dictionary of resources including storage services
    configs: Configuration for Variable Codebook
    """
```
* **Async:** False
* **Method:** True
* **Class:** VariableCodebook

## args

```python
@property
def args(self):
```
* **Async:** False
* **Method:** True
* **Class:** PromptDecisionTreeNode

## args

```python
@property
def args(self):
```
* **Async:** False
* **Method:** True
* **Class:** PromptDecisionTreeEdge

## class_name

```python
@property
def class_name(self) -> str:
    """
    Get class name for this service
    """
```
* **Async:** False
* **Method:** True
* **Class:** VariableCodebook

## execute

```python
def execute(self, action, **kwargs) -> dict[str, Any]:
    """
    Execute variable codebook operations based on the specified action.

This method serves as the primary interface for performing operations on
the variable codebook system. It supports various actions for variable
management, retrieval, and analysis workflow execution. The method delegates
to specific internal handlers based on the action type.

Args:
    action: The operation to perform. Supported actions include variable
        retrieval, prompt sequence generation, assumption access, and
        variable modification operations.
    **kwargs: Action-specific parameters that vary based on the operation
        being performed. Parameter requirements depend on the action type.

Returns:
    dict[str, Any]: Operation results containing success status and relevant
        data. The structure varies based on the action performed but typically
        includes 'success' boolean and operation-specific result data.

Raises:
    ValueError: If the action is not recognized or supported by the system.
    TypeError: If required parameters for the specified action are missing
        or have incorrect types.
    RuntimeError: If the operation fails unexpectedly.

Examples:
    >>> # Execute variable retrieval
    >>> result = codebook.execute("get_variable", variable_name="sales_tax_city")
    >>> 
    >>> # Execute prompt sequence generation
    >>> result = codebook.execute(
    ...     "get_prompt_sequence",
    ...     variable_name="sales_tax_city",
    ...     input_data_point="What is the city tax rate?"
    ... )
    """
```
* **Async:** False
* **Method:** True
* **Class:** VariableCodebook

## get_prompt_sequence_for_input

```python
def get_prompt_sequence_for_input(self, input_data_point: str) -> list[str]:
    """
    Generate a structured prompt sequence for variable extraction from input queries.

This method analyzes an input query or data point to determine the most
relevant variable and returns a sequence of prompts designed to extract
that variable's information. It combines natural language processing with
variable matching to provide intelligent prompt generation for analysis workflows.

Args:
    input_data_point (str): The query, question, or information request
        that needs to be analyzed. This could be a user question about
        tax rates, business requirements, or any variable-related inquiry.

Returns:
    list[str]: Ordered sequence of prompts designed to extract the relevant
        variable information. The prompts are structured to guide systematic
        information extraction from documents or data sources. Returns an
        empty list if no relevant variable can be identified.

Raises:
    TypeError: If input_data_point is not a string.
    ValueError: If input_data_point is empty or contains only whitespace.
    RuntimeError: If variable extraction or prompt generation fails due to
        system errors or service unavailability.

Key Features:
- Intelligent variable identification from natural language queries
- Keyword-based matching with extensible mapping system
- Structured prompt sequence generation based on decision trees
- Support for multiple variable types and categories
- Fallback handling for unrecognized queries

Examples:
    >>> # Tax-related query
    >>> prompts = codebook.get_prompt_sequence_for_input(
    ...     "What is the city sales tax rate for this business?"
    ... )
    >>> print(len(prompts))
    4
    >>> print(prompts[0])
    List the name of the tax as given in the document verbatim...
    >>> 
    >>> # Property tax query
    >>> prompts = codebook.get_prompt_sequence_for_input(
    ...     "How much property tax does the business pay?"
    ... )
    >>> 
    >>> # Unrecognized query
    >>> prompts = codebook.get_prompt_sequence_for_input(
    ...     "What is the weather today?"
    ... )
    >>> print(len(prompts))
    0
    """
```
* **Async:** False
* **Method:** True
* **Class:** VariableCodebook

## run

```python
def run(self, action: str, **kwargs) -> dict[str, Any]:
    """
    Execute variable codebook operations based on the specified action type.

This method provides the core operational interface for the VariableCodebook
system, supporting comprehensive variable management including retrieval,
modification, prompt sequence generation, and assumption access. It uses
pattern matching to route requests to appropriate internal handlers.

Args:
    action (str): The operation to perform. Supported actions include:
        - "get_variable": Retrieve a variable by name
        - "get_prompt_sequence": Generate prompt sequence for variable extraction
        - "get_assumptions": Access assumptions for a specific variable
        - "add_variable": Add a new variable to the codebook
        - "update_variable": Update an existing variable definition
    **kwargs: Action-specific parameters:
        For "get_variable":
            - variable_name (str): Name of the variable to retrieve
        For "get_prompt_sequence":
            - variable_name (str): Name of the variable
            - input_data_point (str): Query or information request
        For "get_assumptions":
            - variable_name (str): Name of the variable
        For "add_variable":
            - variable (Variable): Variable instance to add
        For "update_variable":
            - variable_name (str): Name of variable to update
            - variable (Variable): Updated variable instance

Returns:
    dict[str, Any]: Operation results with the following structure:
        - 'success' (bool): Whether the operation completed successfully
        - Additional keys vary by action type:
            - For get operations: 'variable', 'prompt_sequence', or 'assumptions'
            - For modification operations: 'variable' with updated data
            - For errors: 'error' with descriptive error message

Raises:
    ValueError: If action is not recognized or required parameters are missing.
    TypeError: If parameter types don't match expected types for the action.
    RuntimeError: If operation fails due to system errors or service issues.

Examples:
    >>> # Retrieve a variable
    >>> result = codebook.run("get_variable", variable_name="sales_tax_city")
    >>> if result['success']:
    ...     variable = result['variable']
    >>> 
    >>> # Generate prompt sequence
    >>> result = codebook.run(
    ...     "get_prompt_sequence",
    ...     variable_name="sales_tax_city",
    ...     input_data_point="What is the municipal sales tax rate?"
    ... )
    >>> prompts = result.get('prompt_sequence', [])
    >>> 
    >>> # Add new variable
    >>> new_var = Variable(label="Income Tax", item_name="income_tax", ...)
    >>> result = codebook.run("add_variable", variable=new_var)
    """
```
* **Async:** False
* **Method:** True
* **Class:** VariableCodebook

## serialize_digraph

```python
def serialize_digraph(graph: nx.DiGraph) -> dict:
    """
    Serialize a NetworkX DiGraph into a dictionary format for storage or transmission.

This function converts a NetworkX directed graph into a dictionary representation
that preserves all node attributes, edge attributes, and graph-level attributes.
The resulting dictionary can be used for JSON serialization, database storage,
or transmission between systems.

Args:
    graph (nx.DiGraph): The NetworkX directed graph to serialize.
        Must be a valid DiGraph instance with or without node/edge attributes.

Returns:
    dict: A dictionary containing the serialized graph data with the following structure:
        - nodes (list): List of tuples containing (node_id, attributes_dict)
        - edges (list): List of tuples containing (source, target, attributes_dict)
        - graph_attrs (dict): Dictionary of graph-level attributes

Raises:
    TypeError: If graph is not a NetworkX DiGraph instance.
    AttributeError: If the graph object is missing required NetworkX methods.

Examples:
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_node("A", label="Node A")
    >>> G.add_edge("A", "B", weight=1.5)
    >>> serialized = serialize_digraph(G)
    >>> print(serialized)
    {
        'nodes': [('A', {'label': 'Node A'}), ('B', {})],
        'edges': [('A', 'B', {'weight': 1.5})],
        'graph_attrs': {}
    }
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## validate_digraph

```python
def validate_digraph(v: Any) -> nx.DiGraph:
    """
    Validate and convert input data into a NetworkX DiGraph instance.

This function accepts either an existing NetworkX DiGraph or a dictionary
representation of a graph and ensures it returns a valid DiGraph instance.
It supports deserialization from the format produced by serialize_digraph
and validates the structure of input data.

Args:
    v (Any): Input data to validate and convert. Accepts:
        - nx.DiGraph: Returns the graph as-is after validation
        - dict: Dictionary with 'nodes', 'edges', and 'graph_attrs' keys
        - Other types will raise ValueError

Returns:
    nx.DiGraph: A validated NetworkX directed graph instance with all
        nodes, edges, and attributes properly restored from the input data.

Raises:
    ValueError: If input is not a NetworkX DiGraph or properly formatted
        dictionary, or if dictionary structure is invalid.
    TypeError: If dictionary values are not in the expected format.
    KeyError: If required dictionary keys are missing from input.

Examples:
    >>> # Validate existing DiGraph
    >>> G = nx.DiGraph()
    >>> validated = validate_digraph(G)
    >>> 
    >>> # Validate from dictionary
    >>> graph_dict = {
    ...     'nodes': [('A', {'label': 'Node A'})],
    ...     'edges': [('A', 'B', {'weight': 1.0})],
    ...     'graph_attrs': {'name': 'test_graph'}
    ... }
    >>> validated = validate_digraph(graph_dict)
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A
