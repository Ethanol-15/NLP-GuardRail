import re


class PromptHandler:

    POLICIES = {
        "profanity": "warn",
        "banned_words": "block"
    }

    def __init__(self, regex_rules=None, languages=None):
        """
        Initializes the PromptHandler with specified languages.

        :param languages: List like ["english"], ["filipino"], or ["english", "filipino"]
        """
        self.regex_rules = regex_rules
        self.languages = languages or list(self.regex_rules.keys())

    def detect_matches(self, text: str):
        detections = []

        for lang in self.languages:
            for rule_name, pattern in self.regex_rules[lang].items():
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    detections.append({
                        "language": lang,
                        "rule": rule_name,
                        "match": match.group(),
                    })

        return detections

    def apply_policies(self, detections):
        actions = []

        for detection in detections:
            rule = detection["rule"]
            action = self.POLICIES.get(rule)
            if action:
                actions.append(action)

        return actions

    def analyze_prompt(self, prompt: str):
        detections = self.detect_matches(prompt)
        actions = self.apply_policies(detections)

        return {
            "prompt": prompt,
            "detections": detections,
            "actions": actions,
            "is_safe": len(actions) == 0
        }
