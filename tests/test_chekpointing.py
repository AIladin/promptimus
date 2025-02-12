from promptimus.core.checkpointing import (
    module_dict_from_toml_str,
    module_dict_to_toml_str,
)

MODULE = {
    "params": {
        "censor": "Act as a censor. Detect if user wants to talk about polar bear and return CENSORED. Otherwise return PASS.",
        "moderator": "Ensure compliance with ethical guidelines and return FLAGGED if inappropriate.",
    },
    "submodules": {
        "chat": {
            "params": {"chat": "Act as an assistant"},
            "submodules": {
                "memory": {
                    "params": {
                        "memory": "Store user messages for better contextual responses"
                    },
                    "submodules": {
                        "long_term": {
                            "params": {
                                "long_term": "Maintain session history across interactions"
                            },
                            "submodules": {
                                "vector_store": {
                                    "params": {
                                        "embedding": "Use vector embeddings for semantic search",
                                        "retrieval": "Retrieve relevant past interactions",
                                    },
                                    "submodules": {
                                        "indexer": {
                                            "params": {
                                                "algorithm": "HNSW indexing for fast retrieval",
                                                "dimension": "768-dimensional embeddings",
                                            },
                                            "submodules": {},
                                        }
                                    },
                                }
                            },
                        },
                        "short_term": {
                            "params": {
                                "short_term": "Track recent messages for context within a session"
                            },
                            "submodules": {},
                        },
                    },
                },
                "response_filter": {
                    "params": {
                        "filter": "Analyze responses for policy violations before sending"
                    },
                    "submodules": {
                        "toxicity_detector": {
                            "params": {
                                "model": "Detect toxic language and return SCORE",
                                "threshold": "0.75 for intervention",
                            },
                            "submodules": {},
                        }
                    },
                },
            },
        },
        "analytics": {
            "params": {"tracking": "Log user interactions for system improvement"},
            "submodules": {
                "metrics": {
                    "params": {
                        "latency": "Measure response time",
                        "accuracy": "Track model performance",
                    },
                    "submodules": {
                        "dashboard": {
                            "params": {
                                "visualization": "Provide a UI for monitoring metrics"
                            },
                            "submodules": {},
                        }
                    },
                }
            },
        },
    },
}

TOML_STRING = '''
censor = """
Act as a censor. Detect if user wants to talk about polar bear and return CENSORED. Otherwise return PASS.
"""

moderator = """
Ensure compliance with ethical guidelines and return FLAGGED if inappropriate.
"""


[chat]
chat = """
Act as an assistant
"""


[chat.memory]
memory = """
Store user messages for better contextual responses
"""


[chat.memory.long_term]
long_term = """
Maintain session history across interactions
"""


[chat.memory.long_term.vector_store]
embedding = """
Use vector embeddings for semantic search
"""

retrieval = """
Retrieve relevant past interactions
"""


[chat.memory.long_term.vector_store.indexer]
algorithm = """
HNSW indexing for fast retrieval
"""

dimension = """
768-dimensional embeddings
"""


[chat.memory.short_term]
short_term = """
Track recent messages for context within a session
"""


[chat.response_filter]
filter = """
Analyze responses for policy violations before sending
"""


[chat.response_filter.toxicity_detector]
model = """
Detect toxic language and return SCORE
"""

threshold = """
0.75 for intervention
"""


[analytics]
tracking = """
Log user interactions for system improvement
"""


[analytics.metrics]
latency = """
Measure response time
"""

accuracy = """
Track model performance
"""


[analytics.metrics.dashboard]
visualization = """
Provide a UI for monitoring metrics
"""
'''


def test_to_toml():
    generated_toml_str = module_dict_to_toml_str(MODULE)
    assert generated_toml_str.strip("\n") == TOML_STRING.strip("\n")


def test_from_toml():
    generated_module_dict = module_dict_from_toml_str(TOML_STRING)
    assert generated_module_dict == MODULE
