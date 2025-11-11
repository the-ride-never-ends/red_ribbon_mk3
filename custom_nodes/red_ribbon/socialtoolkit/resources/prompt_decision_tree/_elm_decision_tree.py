"""
ELM decision trees.
"""
import networkx as nx
import logging
from typing import Any, Never, TypeVar


ApiBase = TypeVar('ApiBase')


logger = logging.getLogger(__name__)


def log_then_raise(exc: Exception, msg: str) -> Never:
    """Log a message then raise an exception associated with it."""
    logger.error(msg)
    raise exc(msg)

class DecisionTree:
    """Class to traverse a directed graph of LLM prompts. Nodes are
    prompts and edges are transitions between prompts based on conditions
    being met in the LLM response."""

    def __init__(self, graph):
        """Class to traverse a directed graph of LLM prompts. Nodes are
        prompts and edges are transitions between prompts based on conditions
        being met in the LLM response.

        Examples
        --------
        Here's a simple example to setup a decision tree graph and run with the
        DecisionTree class:

        >>> import logging
        >>> import networkx as nx
        >>> from rex import init_logger
        >>> from elm.base import ApiBase
        >>> from elm.tree import DecisionTree
        >>>
        >>> init_logger('elm.tree')
        >>>
        >>> G = nx.DiGraph(text='hello', name='Grant',
                           api=ApiBase(model='gpt-35-turbo'))
        >>>
        >>> G.add_node('init', prompt='Say {text} to {name}')
        >>> G.add_edge('init', 'next', condition=lambda x: 'Grant' in x)
        >>> G.add_node('next', prompt='How are you?')
        >>>
        >>> tree = DecisionTree(G)
        >>> out = tree.run()
        >>>
        >>> print(tree.all_messages_txt)

        Parameters
        ----------
        graph : nx.DiGraph
            Directed acyclic graph where nodes are LLM prompts and edges are
            logical transitions based on the response. Must have high-level
            graph attribute "api" which is an ApiBase instance. Nodes should
            have attribute "prompt" which is a string that can have {format}
            named arguments that will be filled from the high-level graph
            attributes. Nodes can also have "callback" attributes that are
            callables that act on the LLM response in an arbitrary way. The
            function signature for a callback must be
            ``callback(llm_response, decision_tree, node_name)``.
            Edges can have attribute "condition" that is a callable to be
            executed on the LLM response text that determines the edge
            transition. An edge from a node without a condition acts as an
            "else" statement if no other edge conditions are satisfied. A
            single edge from node to node does not need a condition.
        """
        self._graph: nx.DiGraph = graph
        self._history: list = []
        assert isinstance(self.graph, nx.DiGraph)
        assert 'api' in self.graph.graph

    def __getitem__(self, key: Any) -> Any:
        """Retrieve a node by name (str) or edge by (node0, node1) tuple"""
        out = None
        # Look in nodes first
        if key in self.graph.nodes:
            out = self.graph.nodes[key]
        # Then look in edges
        elif key in self.graph.edges:
            out = self.graph.edges[key]
        else:
            log_then_raise(KeyError, f'Could not find "{key}" in graph')
        return out

    @property
    def api(self) -> ApiBase:
        """Get the ApiBase object."""
        return self.graph.graph['api']

    @property
    def messages(self) -> list[str]:
        """List of conversation messages with the LLM."""
        return self.api.messages

    @property
    def all_messages_txt(self) -> str:
        """Printout of the full conversation with the LLM"""
        return self.api.all_messages_txt

    @property
    def history(self) -> list[str]: # ???
        """Record of the nodes traversed in the tree"""
        return self._history

    @property
    def graph(self) -> nx.DiGraph:
        """A networkx graph object"""
        return self._graph

    def call_node(self, node_name: str) -> str:
        """Call the LLM with the prompt from the input node and search the
        successor edges for a valid transition condition

        Args:
            node_name (str): Name of node being executed

        Returns
            out (str): Next node or LLM response if at a leaf node.
        """
        node = self[node_name]
        prompt = self._prepare_graph_call(node_name)
        out: str = self.api.chat(prompt)
        node['response'] = out

        if 'callback' in node:
            callback = node['callback']
            callback(out, self, node_name)

        return self._parse_graph_output(node_name, out)

    def _prepare_graph_call(self, node_name):
        """Prepare a graph call for given node."""
        # Get the prompt from the node
        prompt = self[node_name]['prompt']

        # Fill in the prompt with the graph attributes
        txt_fmt = {k: v for k, v in self.graph.graph.items() if k != 'api'}

        # Fill in the prompt with the node attributes
        prompt = prompt.format(**txt_fmt)

        self._history.append(node_name)
        return prompt

    def _parse_graph_output(self, node0: str, out: str) -> str:
        """Parse graph output for given node and LLM call output. """
        # Get successors and edges
        successors = list(self.graph.successors(node0))
        edges = [self[(node0, node1)] for node1 in successors]
        conditions = [edge.get('condition', None) for edge in edges]

        # If no successors, we are at a leaf node
        if len(successors) == 0:
            logger.info(f'Reached leaf node "{node0}".')
            return out

        # If only one successor and no conditions, raise error.
        if len(successors) > 1 and all(c is None for c in conditions):
            log_then_raise(AttributeError, f"At least one of the edges from '{node0}' should have a condition: {edges}")

        # prioritize callable conditions
        for i, condition in enumerate(conditions):
            if callable(condition) and condition(out):
                logger.info(f'Node transition: "{node0}" -> "{successors[i]}" '
                            '(satisfied by callable condition)')
                return successors[i]

        # None condition is basically "else" statement
        for i, condition in enumerate(conditions):
            if condition is None:
                logger.info(f'Node transition: "{node0}" -> "{successors[i]}" '
                            '(satisfied by None condition)')
                return successors[i]

        log_then_raise(AttributeError, f"None of the edge conditions from '{node0}' were satisfied: {edges}")

    def run(self, node0: str = 'init') -> str:
        """Traverses the decision tree starting at the input node.

        Args:
            node0: Name of starting node in the graph. Defaults to 'init'.

        Returns:
            Final response from LLM at the leaf node.

        Raises:
            RuntimeError: If an exception occurs during tree traversal.
        """
        self._history = []

        while True:
            try:
                out: str = self.call_node(node0)
            except Exception as e:
                last_message = self.messages[-1]['content']
                msg = ('Ran into an exception when traversing tree. '
                       'Last message from LLM is printed below. '
                       'See debug logs for more detail. '
                       '\nLast message: \n'
                       f'"""\n{last_message}\n"""')
                logger.debug('Error traversing trees, heres the full '
                             'conversation printout:'
                             f'\n{self.all_messages_txt}')
                logger.error(msg)
                raise RuntimeError(msg) from e
            # If the outputs are in the graph, call the next node with those outputs
            if out in self.graph:
                node0 = out
            else:
                break

        logger.info(f'Output: {out}')

        return out
