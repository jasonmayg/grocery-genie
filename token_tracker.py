import tiktoken

def track_tokens(text: str):
    """
    Counts tokens for a given text and estimates cost.
    Cost is a demo estimate (not charged in mock mode).
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(text))
    estimated_cost = round((tokens / 1000) * 0.002, 4)  # demo pricing
    return tokens, estimated_cost
