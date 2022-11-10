from keybert import KeyBERT

doc = """
Face the winds of fall and winter with our reflective striped jacket with a graphic lettering design. 
A versatile layering piece, it can be worn with a matching tracksuit and sneakers for a sporty casual look, 
or elevated with denim and combat boots for that coveted gopnik punk aesthetic. 
"""

model = KeyBERT()
keywords = model.extract_keywords(docs=doc, top_n=5, stop_words=None, keyphrase_ngram_range=(1, 1))

print(keywords)