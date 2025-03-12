
# Plug-in-Play Transformer
# By Kyle Rose, Claude 3.7 Sonnet, Codestral
# Version 0.0.1

Description: Create your own custom Transformer implementation in ComfyUI.
This library has the following: # TODO Finish writing a descriptive table of contents.
1. attention.py
- 
-
-
- 
2. mlp.py
3. add_residual.py
4. lay_normalization.py
5. transformer_block.py
6. architecture_explorer.py
7. practical_implementation.py
8. custom_math.py
9. tensor_transformations.py
10. latex_to_python.py
11. plug_in_play_transformer.py
- 



# Custom Function Based Transformer

graph TD
    Input[Input Tensor] --> Embed[Embedding Layer]
    Embed --> LN1[Layer Norm]
    
    LN1 --> QKV[QKV Projection]
    QKV --> |q,k,v| CustomAtt[Custom Attention Function Node]
    CustomAtt --> AttProj[Attention Projection]
    
    Input --> |Residual| Add1[Add]
    AttProj --> Add1
    
    Add1 --> LN2[Layer Norm]
    LN2 --> CustomMLP[Custom MLP Function Node]
    
    Add1 --> |Residual| Add2[Add]
    CustomMLP --> Add2
    
    FnLib[Function Library Node] -.-> |"Provides Code"| CustomAtt
    FnLib -.-> |"Provides Code"| CustomMLP
    
    MathExpr[Mathematical Expression Node] -.-> |"Modifies Function"| CustomAtt
    
    Add2 --> Output[Output]


# LaTeX to Python Code Pipeline

graph TD
    PaperImage[Research Paper Image] --> VisionLLM[Vision LLM]
    VisionLLM --> LaTeXExtractor[LaTeX Equation Extractor]
    LaTeXExtractor --> |Extracted LaTeX| Translator[LaTeX to Python Translator]
    
    Translator --> |Raw Python Code| CodeProcessor[Code Processor]
    CodeProcessor --> |Formatted Python| ComfyNode[ComfyUI Node Generator]
    
    LaTeXExtractor --> |Equation Context| LLMTranslator[LLM-Assisted Translator]
    LLMTranslator --> |Optimized Code| CodeProcessor
    
    VariableMappings[Variable Mappings] --> Translator
    EquationContext[Equation Context] --> LLMTranslator
    
    ComfyNode --> |Custom Attention Node| TransformerIntegration[Transformer Integration]
    ComfyNode --> |Custom MLP Node| TransformerIntegration
    
    TransformerIntegration --> FinalImpl[Final Implementation]
    
    subgraph "Extraction Phase"
        VisionLLM
        LaTeXExtractor
    end
    
    subgraph "Translation Phase"
        Translator
        LLMTranslator
        VariableMappings
        EquationContext
    end
    
    subgraph "Integration Phase"
        CodeProcessor
        ComfyNode
        TransformerIntegration
    end