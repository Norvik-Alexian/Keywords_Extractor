import numpy as np

from transformers import TokenClassificationPipeline, AutoModelForTokenClassification, AutoTokenizer
from transformers.pipelines import AggregationStrategy


class KeyphraseExtractor(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs, **kwargs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE
        )

        return np.unique([result.get('word').strip() for result in results])


model = 'ml6team/keyphrase-extraction-kbir-inspec'
extractor = KeyphraseExtractor(model=model)

text = '''
Keyphrase extraction is a technique in text analysis where you extract the
important keyphrases from a document. Thanks to these keyphrases humans can
understand the content of a text very quickly and easily without reading it
completely. Keyphrase extraction was first done primarily by human annotators,
who read the text in detail and then wrote down the most important keyphrases.
The disadvantage is that if you work with a lot of documents, this process
can take a lot of time.
'''

print(extractor(text))