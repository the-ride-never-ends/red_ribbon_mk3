"""
Plug-in-Play Transformer - Build and train LLMs without knowing how to code!
"""

class TransformerAPI:
    """API for accessing Plug-in-Play Transformer functionality from ComfyUI"""
    
    def __init__(self, resources, configs):
        self.configs = configs
        self.resources = resources

# Main function that can be called when using this as a script
def main():
    print("Plug-in-Play Transformer module loaded successfully")
    print("Available tools:")
    print("- PiPTransformerNode: Node for ComfyUI integration")
    print("- TransformerAPI: API for programmatic access")

if __name__ == "__main__":
    main()