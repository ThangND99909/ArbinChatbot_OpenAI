import json
import os
from ai_core.utils.text_normalizer import remove_accents

class KnowledgeBase:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.rules = []
        self.fallback = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Knowledge base file not found: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.rules = data.get("rules", [])
        self.fallback = data.get("fallback", {})

    def find_answer(self, question: str, lang: str = "en"):
        q_norm = remove_accents(question.lower())

        for rule in self.rules:
            keys = [remove_accents(k.lower()) for k in rule.get("keywords", [])]
            if any(k in q_norm for k in keys):
                return rule.get(lang, rule.get("en", ""))

        # fallback
        return self.fallback.get(lang, self.fallback.get("en", ""))
