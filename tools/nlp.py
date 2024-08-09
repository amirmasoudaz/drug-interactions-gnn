import spacy


class SpaCy:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the NLP tools.
        :param spacy_model:  str: spaCy model to use.
        """
        if spacy_model not in spacy.util.get_installed_models():
            spacy.cli.download(spacy_model)
        self.spacy = spacy.load(spacy_model)

    def preprocess_text(self, text):
        return ' '.join([
            token.text.lower()
            for token in self.spacy(text)
            if not token.is_space
        ])

    def deep_preprocess_text(self, text):
        return ' '.join([
            token.lemma_.lower()
            for token in self.spacy(text)
            if not (token.is_punct | token.is_space | token.is_stop)
        ])
