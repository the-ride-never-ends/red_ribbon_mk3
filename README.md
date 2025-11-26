# Socialtoolkit
## Turn Law into Datasets
### Author: Kyle Rose, various LLMs 
### Version: 0.1


## Overview

Socialtoolkit is a custom data pipeline that automates legal research and dataset creation.
It gets documents from trusted sources such as state government websites, processes them through tailored data extraction pipelines, and uses them to answer user queries. The system is designed to provide accurate, context-aware responses by combining document retrieval with intelligent relevance assessment.

## Getting Started

### Requirements
- Python 3.12 or higher
- Required Python packages (see `requirements.txt`, `requirements_custom_nodes.txt`, )
- An OpenAI API key and SQL database service
- Pre-approved source documents or URLs to legal texts

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   bash install.sh
   ```
3. Configure your document database
4. Set up your database and LLM API credentials in `configs.yaml` and `private_configs.yaml`

## Usage

### Basic Operation

1. **Configure Document Sources**: Add URLs to pre-approved law document sources in the configuration file `configs.yaml`.

2. **Query the System**: Input your query (e.g., "What is the local sales tax in Cheyenne, WY?")

3. **Receive Response**: The system retrieves relevant documents, assesses their relevance using the LLM, and generates a response based on the most pertinent information.

### Example Usage - API

```python
# Example: Querying for specific information
query = "What is the local sales tax in Cheyenne, WY?"
socialtoolkit = make_socialtoolkit_api()
result = socialtoolkit.execute(query)
print(result)
```

### Example Usage - CLI

```bash
python socialtoolkit.py --query "What is the local sales tax in Cheyenne, WY?"
```

## Output Format
```json
{
  "query": "What is the local sales tax in Cheyenne, WY?",
  "answer": "6%",
  "confidence": 0.95,
  "sources": [
    {
        "citation_1": "Wyo. Dep't of Revenue, Sales and Use Tax, https://revenue.wyo.gov/divisions/excise-tax-division/sales-and-use-tax (last visited Jan. 15, 2024)",
        "relevance": 0.98
    },
    {
        "citation_2": "Cheyenne City Code, Chapter 4.16 - Sales and Use Tax, https://library.municode.com/wy/cheyenne/codes/code_of_ordinances?nodeId=COOR_CH4.16SALAUTAX (last visited Jan. 15, 2024)",
        "relevance": 0.92
    }
  ]
}
```

## Troubleshooting

### Common Issues

**Documents not retrieving:**
- Check that URLs are in the approved sources list
- Verify network connectivity
- Ensure URLs are accessible

**Low relevance scores:**
- Adjust similarity threshold in configuration
- Review document sources for relevance
- Consider expanding approved document sources

**LLM errors:**
- Verify API credentials
- Check API rate limits
- Ensure sufficient token allocation

## Best Practices

1. **Source Management**: Regularly review and update approved document sources
2. **Query Formulation**: Be specific with the information you want and the location(s) in queries for better results
3. **Result Validation**: Always verify information from the pipeline before using it in critical decision-making


## Version History
- v0.1: Initial public repo