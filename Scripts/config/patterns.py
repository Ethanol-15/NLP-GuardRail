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
