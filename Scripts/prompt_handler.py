import re


class PromptHandler:

    REGEX_RULES = {
        "english": {
            "profanity": (
                r"\b("
                r"damn(ed|ing)?|"
                r"hell|"
                r"crap(py)?|"
                r"bloody|"
                r"piss(ed|ing)?|"
                r"ass(hole|hat|wipe|clown)?|"
                r"bastard(s)?|"
                r"douche(bag)?|"
                r"shit(s|ty|head|ting)?|"
                r"bullshit(ting)?|"
                r"jackass(es)?|"
                r"dumbass(es)?|"
                r"f[\*@#u]?ck(er|ing|ed|s)?|"
                r"motherf[\*@#u]?cker(s)?|"
                r"dick(s)?|"
                r"prick(s)?|"
                r"twat(s)?|"
                r"bitch(es|y)?"
                r")\b"
            ),

            "banned_words": (
                r"\b("
                r"porn(ography|o)?|"
                r"sex(ual)?|"
                r"blow(job)?|"
                r"hand(job)?|"
                r"gangbang(s)?|"
                r"dildo(s)?|"
                r"vibrator(s)?|"
                r"semen|"
                r"cum(ming|shot)?|"
                r"orgasm(s)?|"
                r"clitoris|"
                r"vagina(s)?|"
                r"penis(es)?|"
                r"anal|"
                r"rim(job)?|"
                r"deepthroat(s)?|"
                r"tit(s|ties)?|"
                r"nudity"
                r")\b"
            ),
        },

        "filipino": {
            "profanity": (
                r"\b("
                r"gago|gaga|ulol|tanga|bobo|"
                r"puta(ng[\s-]?ina)?|"
                r"pakshet|bwisit|lintik|leche|"
                r"yawa|demonyo|punyeta|"
                r"tarantado|hinayupak|"
                r"siraulo|kupal|tite|titi"
                r")\b"
            ),

            "banned_words": (
                r"\b("
                r"kantot(an)?|"
                r"jakol(an)?|"
                r"finger|"
                r"chupa(han)?|"
                r"blow(job)?|"
                r"talsik|tamod|"
                r"puke|pekpek|"
                r"ari|burat|titi|"
                r"pwet(in)?|"
                r"laglag[\s-]?panty|"
                r"nude|hubad|hubo(t)?"
                r")\b"
            ),
        }
    }

    POLICIES = {
        "profanity": "warn",
        "banned_words": "block"
    }

    def __init__(self, languages=None):
        """
        Initializes the PromptHandler with specified languages.

        :param languages: List like ["english"], ["filipino"], or ["english", "filipino"]
        """
        self.languages = languages or list(self.REGEX_RULES.keys())

    def detect_matches(self, text: str):
        detections = []

        for lang in self.languages:
            for rule_name, pattern in self.REGEX_RULES[lang].items():
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
