

import openai



class OpenAiApi: # TODO
    """
    Wrapper for OpenAI API
    """

    def __init__(self, resources, configs):
        self.configs = configs
        self.resources = resources or {}

        self.api_key = self.configs.api_key
        self.api = openai.API(self.api_key)

