import re


class PromptHandler:
    """
    Class for handling and analyzing prompts for profanity and banned words in English and Filipino.

    Attributes:
        regex_rules (dict): A dictionary containing regex patterns for different languages and rules.
        languages (list): List of languages to analyze prompts in.
    """

    POLICIES = {
        "profanity": "warn",
        "banned_words": "block"
    }

    def __init__(self, regex_rules=None, languages=None) -> None:
        """
        Initializes the PromptHandler with specified languages.

        :param languages: List like ["english"], ["filipino"], or ["english", "filipino"]
        """
        self.regex_rules = regex_rules
        self.languages = languages or list(self.regex_rules.keys())

    def detect_matches(self, text: str) -> list:
        """
        Detects matches in the given text based on the regex rules for the specified languages.

        Args:
            text (str): The input text to analyze.

        Returns:
            list: A list of detected matches with their corresponding language and rule.
        """
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

    def apply_policies(self, detections: list) -> list:
        """
        Applies policies based on detected matches.

        Args:
            detections (list): List of detected matches.

        Returns:
            list: List of actions to be taken based on the detections.
        """
        actions = []

        for detection in detections:
            rule = detection["rule"]
            action = self.POLICIES.get(rule)
            if action:
                actions.append(action)

        return actions

    def analyze_prompt(self, prompt: str) -> dict:
        """
        Analyzes the given prompt for profanity and banned words.

        Args:
            prompt (str): The input prompt to analyze.

        Returns:
            dict: A dictionary containing the prompt, detections, actions, and safety status.
        """
        detections = self.detect_matches(prompt)
        actions = self.apply_policies(detections)

        return {
            "prompt": prompt,
            "detections": detections,
            "actions": actions,
            "is_safe": len(actions) == 0
        }
