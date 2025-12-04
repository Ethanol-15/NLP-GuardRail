import re

PATTERN_RULES = {
    "profanity": [
        r"\b(gago|puta|ulol|tanga)\b",
    ],
    "pii_risk": [
        r"\b\d{11}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ],
}

POLICIES = {
    "profanity": "warn",
    "pii_risk": "block",
}


def detect_matches(text: str):
    """
    Returns a list of (rule_name, matched_text) detections.
    """
    detections = []

    for rule_name, patterns in PATTERN_RULES.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                detections.append((rule_name, match.group()))

    return detections


def apply_policies(text: str):
    """
    Applies the policy actions based on what rules the text triggers.
    """
    detections = detect_matches(text)
    actions = []

    for rule_name, _ in detections:
        action = POLICIES.get(rule_name)
        if action:
            actions.append(action)

    return actions


def analyze_prompt(prompt: str):
    detections = detect_matches(prompt)
    actions = apply_policies(prompt)

    return {
        "prompt": prompt,
        "detections": detections,
        "actions": actions,
        "is_safe": len(actions) == 0
    }


prompts = [
    "Sinong tanga gumawa nito?",
    "Aba gago ka!",
    "Nakipagkita ako kay mama kahapon",
    "Nagpunta ako sa paaralan"
]

for prompt in prompts:
    print(analyze_prompt(prompt))
    print()
