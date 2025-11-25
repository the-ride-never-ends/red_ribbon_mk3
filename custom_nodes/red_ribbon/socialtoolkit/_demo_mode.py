"""
Demo functions for Socialtoolkit nodes.
"""
import random
import sys
import time
from typing import Any, Optional, Type


import comfy.utils # type: ignore


try:
    from ..utils_.common._safe_format import safe_format
    from ..custom_easy_nodes import show_text
except ImportError as e:

    def safe_format(format_string: str, *args: tuple, **kwargs: dict) -> str:
        return format_string.format(**kwargs)

    def show_text(text: str) -> None:
        print(text)


_LLM_MODELS: list[str] = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "gpt-4o"]

# temperature: float = NumberInput(default=0.7, min=0.1, max=1.0, step=0.1),
# max_tokens: int = NumberInput(default=4096, min=1, max=10000, step=1),
# top_p: float = NumberInput(default=1.0, min=0.1, max=1.0, step=0.1),


_INPUT_TEXT_OPTIONS: list[str] = [
    "Local Sales Tax in Cheyenne, WY",
    "What is the local sales tax in Springhill, Lousiana?",
    "List exceptions to local vending machine laws in San Jose, CA",
]

# 37
_DEMO_INPUT_URLS: list[str] = [
    # Springhill, LA = Query "Local Sales Tax in Springhill, LA"
    "https://library.municode.com/la/springhill/codes/code_of_ordinances",
    # Cheyenne, WY = Query "Local Sales Tax in Cheyenne, WY"
    "https://library.municode.com/wy/cheyenne/codes/code_of_ordinances",
    # San Jose, CA = Query "Vending Machine Laws in San Jose, CA",
    "https://library.municode.com/ca/san_jose/codes/code_of_ordinances",
]

_DEMO_RELEVANT_URLS: dict[str, list[str]] = {
    # Springhill, LA
    "springhill": [
        "https://library.municode.com/la/springhill/codes/code_of_ordinances?nodeId=COOR_CH98TA_ARTIINGE_S98-1SAUSTA",
        "https://taxfoundation.org/location/louisiana/",
        "https://webstersalestax.org/current-rates/"
    ],
    # Cheyenne, WY
    "cheyenne": [
        "https://library.municode.com/wy/cheyenne/codes/code_of_ordinances?nodeId=TIT3REFI_CH3.08TA_3.08.010REEXINRE",
        "https://taxfoundation.org/location/wyoming/",
        "https://www.avalara.com/taxrates/en/state-rates/wyoming/counties/laramie-county.html",
        "https://www.cityoflaramie.org/FAQ.aspx?QID=104",
    ],
    # San Jose, CA
    "san_jose": [
        "https://library.municode.com/ca/san_jose/codes/code_of_ordinances?nodeId=TIT20ZO_CH20.80SPUSRE_PT10OUVEFA_20.80.820EXDMPE"
        "https://library.municode.com/ca/san_jose/codes/code_of_ordinances?nodeId=TIT6BULIRE_CH6.54PEPEOR",
    ],
}

ANSWER: str = "6%"  # Default answer
_VECTOR_LENGTH: int = 3072 # NOTE: See: https://platform.openai.com/docs/guides/embeddings
_SELECTED_DOMAIN_URLS: list[str] = []


def _mock_internet_check(links: list) -> int:
    print("Getting laws from the web...")
    #random_sleep()
    print("Checking government websites...")
    #random_sleep()
    print("Checking Google...")
    #random_sleep()
    print("Checking Bing...")
    random_sleep()

    len_domain_urls = len(links)
    random_number_of_urls = len_domain_urls * round(random.uniform(0, 5))
    print(f"Found {random_number_of_urls} laws. Downloading...")
    return random_number_of_urls


def flatten_nested_dictionary_into_list(input_dict: dict) -> list:
    output = []
    for key, value in input_dict.items():
        if isinstance(value, list):
            output.extend(value)
        else:
            if isinstance(value, dict):
                flatten_nested_dictionary_into_list(value)
            else:
                output.append(value)
    return output


def random_sleep(start: Optional[int] = None, stop: Optional[int] = None) -> None:
    if start is None or stop is None:
        start, stop = 1, 3
    rand_int = random.randint(start, stop)
    time.sleep(rand_int)


def _stream_llm_response(llm: dict, func_name: str, **kwargs) -> None:
    print(f"llm_dict:\n{llm}")
    messages = llm['ai']['messages'][func_name][1]["content"]
    print(f"messages: {messages}")
    if kwargs:
        llm['ai']['messages'][func_name][1]["content"] = safe_format(messages, **kwargs)
    collected_chunks = []
    collected_messages = []
    response = llm['ai']['client'].chat.completions.create(
        model=llm["model"],
        messages=llm['ai']['messages'][func_name][1]["content"],
        temperature=0.3,
        stream=True
    )
    timer_wait = 0.01
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print_chunk = []
        for message in collected_messages:
            if message is None:
                break
            print_chunk.append(message)
            flattened_chunk = "".join(print_chunk)
            sys.stdout.write('\r')
            sys.stdout.write(flattened_chunk)
            sys.stdout.flush()
            time.sleep(timer_wait)


def demo_load_data() -> str:
    print("Loading links...")
    random_sleep(1,2)
    urls = "mock_urls"
    print("Links loaded successfully.")
    return urls


def demo_document_retrieval_from_websites(links: list) -> dict:
    mock_vectors = []
    random_number_of_urls = _mock_internet_check(links)
    pbar = comfy.utils.ProgressBar(total=random_number_of_urls)
    for i,_ in enumerate([random_number_of_urls], start=1):
        mock_vectors.append([random.uniform(-5.0, 5.0) for _ in range(_VECTOR_LENGTH)])
        time.sleep(0.1)
        #random_sleep(1,2)
        print(f"Downloading law {i}.")
        if i == 1:
            pbar.update(1)
        else:
            pbar.update(i+1)
    print("Laws downloaded. Printing...")
    link_dict = flatten_nested_dictionary_into_list(_DEMO_RELEVANT_URLS)
    for link in link_dict:
        print(link)

    return {
        "documents": _SELECTED_DOMAIN_URLS,
        "metadata": "mock_metadata",
        "vectors": mock_vectors
    }


def demo_document_storage() -> dict:
    print("Saving laws to disk...")
    random_sleep(2,5)
    print("Laws saved.")
    return {
        "documents": [f"mock_document {i}" for i in range(10)],
        "metadata": "mock_metadata",
        "vectors": [[random.uniform(-5.0, 5.0) for _ in range(_VECTOR_LENGTH)] for _ in range(10)]
    }

def demo_top10_document_retrieval(laws: list, return_how_many: int) -> list:
    print(f"Filtering down to {return_how_many} documents...")
    print("Laws filtered successfully.")
    return laws

def demo_relevance_assessment(ai: Any, laws: dict) -> str:
    print("Filtering laws with AI...")
    random_sleep(1,2)
    print("Laws filtered successfully. Printing results:")
    question: str = ai['question']
    documents: list[str] = []
    match question:
        case question if "Cheyenne" in question:
            documents = _DEMO_RELEVANT_URLS['cheyenne']
        case question if "Springhill" in question:
            documents = _DEMO_RELEVANT_URLS['springhill']
        case question if "San Jose" in question:
            documents = _DEMO_RELEVANT_URLS['san_jose']
        case _: # Default case is to return all the documents
            sub_dicts = _DEMO_RELEVANT_URLS.values()
            for doc in sub_dicts:
                documents.extend(doc)
    #_stream_llm_response(ai, "relevance_assessment", relevant_docs=documents)
    print(f"#########\n{documents}")
    laws['documents'] = documents
    return question

def demo_database_enter() -> str:
    print("Logging into database...")
    random_sleep()
    database_api = "mock_db"
    print("Login successful.")
    return database_api

def demo_prompt_decision_tree() -> str:
    show_text("AI is reviewing the laws...")
    random_sleep(1)
    show_text("Law review complete. Answering question...")
    random_sleep(1)
    print("Question answered.")
    answer: str = "blank"
    return answer

def demo_variable_codebook(question: str, documents: list[str]) -> str:
    global ANSWER
    print("Loading AI instructions...")
    random_sleep(2,3)
    match question:
        case question if "Cheyenne" in question:
            print("Found applicable instructions: Local Sales Tax.")
            ANSWER = answer = "6.0%"
        case question if "Springhill" in question:
            print("Found applicable instructions: Local Sales Tax.")
            ANSWER = answer = "10.5%"
        case question if "San Jose" in question:
            print("Found applicable instructions: Vending Machine Laws.")
            ANSWER = answer = "Exceptions for Holiday sales, city-approved locations, newsracks, on-site businesses, and farmers' markets."
        case _:
            ANSWER = answer = "mock_prompts"
    print("AI instructions loaded.")
    return answer


def demo_llm_api(name: str, instructions: str, red_ribbon: Type) -> dict:
    print(f"AI model '{name}' selected.")
    print("Loading AI...")
    random_sleep(2,3)
    print(f"{name} loaded.")
    
    llm = {
        "model": name,
        "answer": instructions[0],
        "instructions": instructions[1],
        "question": instructions[1],
        "prompts": instructions[1],
        "ai": {
            "client": red_ribbon.client,
            "messages": {
                f'llm_api': [],
                f"relevance_assessment": []
    },}}
    system_prompt: dict = {"role": "system", "content": "You are a helpful assistant.\nYou are currently assisting in a demo presentation for software product.\nSpeak in a stereotypically robotic tone."}
    llm['ai']['messages']["llm_api"] = [
        system_prompt,
        {"role": "user", "content": "You've been initialized. Say hello to the audience!"},
    ]
    llm['ai']['messages']["relevance_assessment"] = [
        system_prompt,
        {"role": "user", "content": "You've found these URLs to be relevant. Present them to the audience.\n###### {relevant_docs}"},
    ]
    return llm

def demo_answer():
    print(f"The answer is: {ANSWER}")
    show_text(ANSWER)


def demo_main(text: str):
    """Main demo function."""
    demo_load_data()
    urls = _DEMO_INPUT_URLS
    demo_document_retrieval_from_websites(urls)
    demo_document_storage()
    demo_database_enter()
    demo_top10_document_retrieval([], 5)
    mock_data = {
        "question": text,
        "documents": [],
    }
    question = demo_relevance_assessment(mock_data,{})
    demo_variable_codebook(question, mock_data['documents'])
    demo_prompt_decision_tree()

    return ANSWER

import argparse
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo Socialtoolkit")
    parser.add_argument(
        "--input_text",
        type=str,
        default="What is the local sales tax in Springhill, Lousiana?",
        help="Input text for the demo",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        demo_main(args.input_text)
    except Exception as e:
        print(f"An error occurred during the demo: {e}")
