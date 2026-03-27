import tiny_rag_ai

tiny_rag_ai.index("./docs")
print(tiny_rag_ai.chat("How can we actually conquer Proxima B Centuri With current Technologies.", use_case="question assistant on the given topics"))