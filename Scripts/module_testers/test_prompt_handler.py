from Scripts.modules.prompt_handler import PromptHandler
from config.patterns import REGEX_RULES

handler = PromptHandler(regex_rules=REGEX_RULES,
                        languages=["english", "filipino"])

english_prompts = [
    "This is so fucking damn annoying!",
    "What the fuck is this shit?!",
    "I saw a nude picture online",
    "I love programming so much!"
]

filipino_prompts = [
    "Gago ka talaga putanginamo",
    "May nakita akong hubad na tao",
    "Nag-aaral ako ngayon",
    "Sinong tanga gumawa nito?"
]

for text in english_prompts:
    result = handler.analyze_prompt(text)

    print("\n")
    print("Prompt:", result["prompt"])
    print("Is Safe?:", result["is_safe"])
    print("Actions:", result["actions"])

    if result["detections"]:
        print("Detected:")
        for d in result["detections"]:
            print(f"`{d['match']}` ({d['rule']}, {d['language']})")

for text in filipino_prompts:
    result = handler.analyze_prompt(text)

    print("\n")
    print("Prompt:", result["prompt"])
    print("Is Safe?:", result["is_safe"])
    print("Actions:", result["actions"])

    if result["detections"]:
        print("Detected:")
        for d in result["detections"]:
            print(f"`{d['match']}` ({d['rule']}, {d['language']})")
