"""
Module 3 — Transliteration / Normalization (Pre-MT Cleanup)
Converts romanized text (Hinglish/Tanglish/Telglish) to native scripts (Devanagari, Tamil, Telugu, etc.)
using indic-transliteration package and deterministic colloquial dictionaries.
"""

from typing import Dict, Any, List, Optional


class TextNormalizer:
    SCRIPT_MAP = {
        "hi": "DEVANAGARI",
        "ta": "TAMIL",
        "te": "TELUGU",
        "kn": "KANNADA",
        "ml": "MALAYALAM",
        "bn": "BENGALI",
        "mr": "DEVANAGARI",
        "gu": "GUJARATI",
        "pa": "GURMUKHI",
        "ur": "URDU"
    }

    # Comprehensive Colloquial Tamil Roman-to-Script Lexicon
    ROMAN_TO_TAMIL = {
        # Pronouns & Addressing
        "yenada": "என்னடா", "ennada": "என்னடா", "yenda": "ஏண்டா", "eanda": "ஏண்டா",
        "yenadi": "ஏண்டி", "ennadi": "என்னடி", "yenpa": "ஏன்பா", "ennapa": "என்னப்பா",
        "nan": "நான்", "naan": "நான்", "nee": "நீ", "neenga": "நீங்கள்", "neengal": "நீங்கள்",
        "unga": "உங்கள்", "ungal": "உங்கள்", "ungala": "உங்களை", "ungalluku": "உங்களுக்கு", "ungalukku": "உங்களுக்கு",
        "yen": "என்", "en": "என்", "ennoda": "என்", "enna": "என்ன", "enakku": "எனக்கு", "enaku": "எனக்கு",
        "unna": "உன்னை", "unoda": "உன்", "unakku": "உனக்கு", "avan": "அவன்", "avana": "அவனை", "avanukku": "அவனுக்கு",
        "aval": "அவள்", "avala": "அவளை", "avalukku": "அவளுக்கு", "avanga": "அவர்கள்", "avangal": "அவர்கள்",
        "avangala": "அவர்களை", "avangalukku": "அவர்களுக்கு", "namma": "நமது", "namakku": "நமக்கு",
        "nanba": "நண்பா", "nanban": "நண்பன்", "machan": "மச்சான்", "machi": "மச்சி", "thambi": "தம்பி", "anna": "அண்ணா",
        "thala": "தலைவா", "thalaiva": "தலைவா", "mama": "மாமா",
        "akka": "அக்கா", "thangachi": "தங்கச்சி", "amma": "அம்மா", "appa": "அப்பா", "sir": "சார்", "boss": "பாஸ்",
        "periyappa": "பெரியப்பா", "periyamma": "பெரியம்மா", "chithappa": "சித்தப்பா", "chithi": "சித்தி",
        "athai": "அத்தை", "athaan": "அத்தான்", "thatha": "தாத்தா", "paati": "பாட்டி",
        "avar": "அவர்", "avaru": "அவர்", "avaruku": "அவருக்கு", "avarukku": "அவருக்கு",
        "friend": "நண்பன்", "friends": "நண்பர்கள்", "bro": "தம்பி",

        # Interrogatives & Demonstratives
        "idhu": "இது", "idha": "இதை", "adhu": "அது", "adha": "அதை", "edhu": "எது",
        "indha": "இந்த", "andha": "அந்த", "enda": "எந்த",
        "smell": "வாசனை", "taste": "சுவை",
        "eppadi": "எப்படி", "epdi": "எப்படி", "enga": "எங்கே", "engae": "எங்கே",
        "inga": "இங்கே", "anga": "அங்கே", "yaaru": "யார்", "yaara": "யாரை", "yaarukku": "யாருக்கு", "eppo": "எப்போது",
        "eppoluthu": "எப்போது", "yean": "ஏன்", "ethana": "எத்தனை", "evvalavu": "எவ்வளவு",
        "edhuku": "எதற்கு", "adhunala": "அதனால்", "adhadhan": "அதைத்தான்", "dhaan": "தான்", "dhan": "தான்",
        "vaah": "வாஹ்", "wah": "வாஹ்", "yosika": "யோசிக்க", "yosikka": "யோசிக்க", "aacha": "ஆச்சா", "neramaacha": "நேரமாச்சா",

        # Time Expressions & Adverbs
        "innaiku": "இன்று", "iinaiku": "இன்று", "nalaiku": "நாளை", "naalaiku": "நாளை",
        "nalaikku": "நாளை", "naalaikku": "நாளை", "nethu": "நேற்று",
        "ippo": "இப்போது", "appo": "அப்போது", "eppovum": "எப்போதும்", "neram": "நேரம்", "seekiram": "சீக்கிரம்",
        "kaalai": "காலை", "kalaila": "காலையில்", "iravu": "இரவு", "medhuva": "மெதுவாக", "marupadiyum": "மீண்டும்",

        # Common Nouns & Places
        "veedu": "வீடு", "veetla": "வீட்டில்", "veetuku": "வீட்டுக்கு", "ooru": "ஊர்", "oorla": "ஊரில்",
        "chennai": "சென்னை", "delhi": "டெல்லி", "mumbai": "மும்பை",
        "saapadu": "உணவு", "sappadu": "உணவு", "thanni": "தண்ணீர்",
        "dinner": "இரவு உணவு", "lunch": "மதிய உணவு", "breakfast": "காலை உணவு", "tiffin": "சிற்றுண்டி",
        "kaasu": "பணம்", "panam": "பணம்", "velai": "வேலை", "padam": "படம்", "cinema": "சினிமா", "movie": "படம்",
        "kavala": "கவலை", "sandai": "சண்டை", "sanda": "சண்டை", "thappu": "தவறு", "sari": "சரி", "seri": "சரி",

        # Quantifiers & Adjectives
        "ellarum": "எல்லோரும்", "ellorum": "எல்லோரும்", "ellam": "எல்லாம்", "konjam": "கொஞ்சம்",
        "romba": "ரொம்ப", "nalla": "நல்ல", "kevalama": "மோசமாக", "super": "சூப்பர்", "periya": "பெரிய",
        "chinna": "சின்ன", "azhaga": "அழகாக", "mukkiyam": "முக்கியம்", "mattum": "மட்டும்",
        "semma": "செம", "semmaya": "செமயா", "mass": "மாஸ்", "gethu": "கெத்து",

        # Common Verbs & Conjugations
        "irukku": "இருக்கிறது", "irukanga": "இருக்கிறார்கள்", "irukken": "இருக்கிறேன்", "irukkom": "இருக்கிறோம்",
        "iruka": "இருக்கிறாய்", "irukka": "இருக்கிறாய்", "irukinga": "இருக்கிறீர்கள்", "irukkeenga": "இருக்கிறீர்கள்",
        "irundhuchu": "இருந்தது", "irundhadhu": "இருந்தது", "illa": "இல்லை", "illai": "இல்லை",
        "poren": "போகிறேன்", "poran": "போகிறான்", "poranga": "போகிறார்கள்", "porom": "போகிறோம்", "ponga": "போங்கள்", "polama": "போலாமா",
        "povala": "போவாளா", "poviya": "போவாயா", "povaaya": "போவாயா", "povaan": "போவான்", "poyiten": "போய்விட்டேன்", "poyiduchu": "போய்விட்டது", "poi": "போய்", "po": "போ",
        "varren": "வருகிறேன்", "varen": "வருகிறேன்", "varan": "வருகிறான்", "varanga": "வருகிறார்கள்",
        "vararu": "வருகிறார்", "varaaru": "வருகிறார்",
        "vaanga": "வாருங்கள்", "varalama": "வரலாமா",
        "varuvala": "வருவாளா", "varuvaala": "வருவாளா", "varuviya": "வருவாயா", "varuvaaya": "வருவாயா", "varuvom": "வருவோம்",
        "vandhu": "வந்து", "vandhutten": "வந்துவிட்டேன்", "vandhuten": "வந்துவிட்டேன்", "vandhurunga": "வந்துவிடுங்கள்", "varapa": "வரும்போது", "vaa": "வா", "va": "வா",
        "vangu": "வாங்கு", "vangitu": "வாங்கிட்டு", "vaangi": "வாங்கி", "vaangitu": "வாங்கிட்டு", "vangidu": "வாங்கிடு", "vangunga": "வாங்குங்கள்",
        "paakanum": "பார்க்க வேண்டும்", "paathen": "பார்த்தேன்", "paaru": "பாரு", "paathingala": "பார்த்தீர்களா", "paakala": "பார்க்கவில்லை", "paathiya": "பார்த்தாயா",
        "solren": "சொல்கிறேன்", "solra": "சொல்கிறார்", "solunga": "சொல்லுங்கள்", "sonna": "சொன்னார்", "sollu": "சொல்", "sonniya": "சொன்னாயா",
        "pesu": "பேசு", "pesathinga": "பேசாதீர்கள்", "pesuranga": "பேசுகிறார்கள்", "pesuren": "பேசுகிறேன்", "pesuvathu": "பேசுவது", "pesuniya": "பேசினாயா",
        "kelu": "கேள்", "kelunga": "கேளுங்கள்", "kekkuren": "கேட்கிறேன்", "kettan": "கேட்டேன்", "ketiya": "கேட்டாயா", "ketkudhu": "கேட்கிறது",
        "kudunga": "கொடுங்கள்", "kudu": "கொடு", "kuduthen": "கொடுத்தேன்", "kuduthiya": "கொடுத்தாயா", "edunga": "எடுங்கள்", "edu": "எடு",
        "pannu": "செய்", "pannunga": "செய்யுங்கள்", "pandriya": "செய்கிறாயா", "pandran": "செய்கிறான்", "panren": "செய்கிறேன்",
        "panra": "செய்கிறாய்", "pandre": "செய்கிறாய்", "panringa": "செய்கிறீர்கள்", "panreenga": "செய்கிறீர்கள்", "pannura": "செய்கிறாய்",
        "pannanum": "நாம் செய்ய வேண்டும்", "pananum": "நாம் செய்ய வேண்டும்",
        "pannalam": "பண்ணலாம்", "panniten": "செய்துவிட்டேன்", "mudiyum": "முடியும்", "mudiyadhu": "முடியாது",
        "mudinjidha": "முடிந்ததா", "mudinjudhu": "முடிந்தது", "theriyum": "தெரியும்",
        "theriyadhu": "தெரியாது", "therila": "தெரியவில்லை", "puriyala": "புரியவில்லை", "puriyadhu": "புரியாது", "puriyudhu": "புரிகிறது", "purinjidha": "புரிந்ததா",
        "theva": "தேவை", "thevapaduthu": "தேவைப்படுகிறது", "pasikudhu": "பசிக்கிறது",
        "adikuthu": "அடிக்குது", "aachu": "ஆனது", "aayidichu": "ஆகிவிட்டது", "aaiduchu": "ஆகிவிட்டது", "aagidum": "ஆகிவிடும்",
        "pannadhume": "செய்ததுமே", "pannadhuma": "செய்ததுமே", "pannadhale": "செய்ததால்",
        "vachuruka": "வைத்திருக்கிறாய்", "vechiruka": "வைத்திருக்கிறாய்",
        "padathenga": "படாதீர்கள்", "padathe": "படாதே", "marandhuten": "மறந்துவிட்டேன்", "serndhu": "சேர்ந்து",
        "pudichirukku": "பிடித்திருக்கிறது", "pudikkala": "பிடிக்கவில்லை", "pudichirundhadhu": "பிடித்திருந்தது",
        "sapdu": "சாப்பிடு", "sapten": "சாப்பிட்டேன்", "saptiya": "சாப்பிட்டாயா", "thoongu": "தூங்கு", "thoongiten": "தூங்கிவிட்டேன்",
        "anuppu": "அனுப்பு", "anupiten": "அனுப்பிவிட்டேன்", "anupunga": "அனுப்புங்கள்", "kandupidi": "கண்டுபிடி",
        "thirumba": "திரும்ப", "marubadiyum": "மறுபடியும்", "kooda": "கூட", "ippave": "இப்போதே", "appove": "அப்போதே",

        # Postpositions, Conjunctions & Particles
        "un": "உன்", "dei": "டேய்", "vanakkam": "வணக்கம்", "nandri": "நன்றி",
        "aama": "ஆம்", "ku": "க்கு", "la": "இல்", "oda": "உடைய", "kitta": "கிட்ட",
        "enkitta": "என்கிட்ட", "unkitta": "உன்கிட்ட", "kuda": "கூட", "kooda": "கூட",
        "ulla": "உள்ள", "veliya": "வெளியே", "mela": "மேலே", "keezha": "கீழே",
        "aana": "ஆனால்", "aanaa": "ஆனால்", "appuram": "அப்புறம்", "piragu": "பிறகு",
        "please": "தயவுசெய்து", "thanks": "நன்றி", "thaan": "தான்", "than": "தான்",
        "dhan": "தான்", "ah": "ஆக", "aaga": "ஆக"
    }

    # Comprehensive Colloquial Hindi Roman-to-Script Lexicon
    ROMAN_TO_DEVANAGARI = {
        # Pronouns & Addressing
        "aap": "आप", "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "aapko": "आपको",
        "tum": "तुम", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी", "tumhare": "तुम्हारे", "tumhe": "तुम्हें",
        "main": "मैं", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "mujhe": "मुझे", "mujhko": "मुझको",
        "hum": "हम", "humara": "हमारा", "humari": "हमारी", "humare": "हमारे", "humein": "हमें",
        "tune": "तूने", "tumne": "तुमने", "maine": "मैंने", "usne": "उसने", "unhone": "उन्होंने", "humne": "हमने",
        "woh": "वह", "wo": "वह", "yeh": "यह", "ye": "यह", "unka": "उनका", "unki": "उनकी", "unke": "उनके",
        "unko": "उनको", "uska": "उसका", "uski": "उसकी", "uske": "उसके", "usko": "उसको",
        "sab": "सब", "log": "लोग", "pura": "पूरा", "poora": "पूरा",
        "bhai": "भाई", "dost": "दोस्त", "yaar": "यार", "behen": "बहन", "mummy": "मम्मी", "papa": "पापा",
        "boss": "बॉस", "sir": "सर", "arrey": "अरे", "arre": "अरे", "wah": "वाह", "waah": "वाह",

        # Interrogatives & Demonstratives
        "kya": "क्या", "kyun": "क्यों", "kyu": "क्यों", "kahan": "कहाँ", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी",
        "kab": "कब", "kaun": "कौन", "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने",

        # Postpositions & Prepositions
        "se": "से", "ko": "को", "ka": "का", "ke": "के", "ki": "की", "mein": "में", "me": "में",
        "par": "पर", "pe": "पे", "tak": "तक", "liye": "लिए", "saath": "साथ", "paas": "पास",
        "bina": "बिना", "andar": "अंदर", "bahar": "बाहर", "upar": "ऊपर", "neeche": "नीचे",
        "aage": "आगे", "peeche": "पीछे", "wala": "वाला", "wali": "वाली", "wale": "वाले",

        # Time Expressions & Frequency
        "aaj": "आज", "kal": "कल", "parson": "परसों", "subah": "सुबह", "shaam": "शाम", "raat": "रात",
        "dopahar": "दोपहर", "abhi": "अभी", "kabhi": "कभी", "phir": "फिर", "baad": "बाद", "hamesha": "हमेशा",

        # Common Nouns & Places
        "kaam": "काम", "naam": "नाम", "ghar": "घर", "pani": "पानी", "paani": "पानी", "khana": "खाना",
        "paisa": "पैसा", "paise": "पैसे", "baat": "बात", "baatein": "बातें", "office": "ऑफिस",
        "dilli": "दिल्ली", "delhi": "दिल्ली", "mumbai": "मुंबई", "chennai": "चेन्नई", "india": "भारत",
        "bhookh": "भूख", "pyaas": "प्यास", "dimaag": "दिमाग", "dil": "दिल", "madad": "मदद", "koshish": "कोशिश",
        "pasand": "पसंद", "smell": "खुशबू", "taste": "स्वाद",

        # Common Verbs & Auxiliaries
        "hai": "है", "hain": "हैं", "hoon": "हूँ", "ho": "हो", "tha": "था", "thi": "थी", "the": "थे",
        "hoga": "होगा", "hogi": "होगी", "honge": "होंगे", "karna": "करना", "karo": "करो", "kar": "कर", "kariye": "करिए",
        "kiya": "किया", "diya": "दिया", "karega": "करेगा", "karenge": "करेंगे", "karta": "करता", "karti": "करती",
        "karte": "करते", "jaana": "जाना", "jao": "जाओ", "ja": "जा", "jaiye": "जाइए", "jaunga": "जाऊंगा", "jaungi": "जाऊंगी",
        "jayega": "जाएगा", "jayenge": "जाएंगे", "gaya": "गया", "gayi": "गयी", "gaye": "गए",
        "aa": "आ", "aana": "आना", "aao": "आओ", "aaiye": "आइए", "aaya": "आया", "aayi": "आयी", "aaye": "आए",
        "aayega": "आएगा", "aayenge": "आएंगे", "raha": "रहा", "rahi": "रही", "rahe": "रहे",
        "dekho": "देखो", "dekh": "देख", "dekha": "देखा", "suno": "सुनो", "sun": "सुन", "suna": "सुना",
        "bolo": "बोलो", "bol": "बोल", "boliye": "बोलिए", "bola": "बोला", "batao": "बताओ", "bata": "बता", "bataya": "बताया",
        "samajh": "समझ", "samjha": "समझा", "chahiye": "चाहिए", "sakta": "सकता", "sakti": "सकती", "sakte": "सकते",
        "dena": "देना", "do": "दो", "dijiye": "दीजिए", "dega": "देगा", "lena": "लेना", "lo": "लो", "lijiye": "लीजिए", "liya": "लिया",
        "sochna": "सोचना", "socho": "सोचो", "socha": "सोचा", "ghoomne": "घूमने", "yaad": "याद",
        "lagaya": "लगाया", "laga": "लगा", "lagi": "लगी", "lagna": "लगना",
        "baje": "बजे", "shuru": "शुरू", "khatam": "खत्म",
        "khelne": "खेलने", "khelna": "खेलना", "khelo": "खेलो", "khel": "खेल",
        "milne": "मिलने", "padhne": "पढ़ने", "dekhne": "देखने", "karne": "करने",

        # Adjectives, Adverbs, Conjunctions & Particles
        "bahut": "बहुत", "bohot": "बहुत", "badhiya": "बढ़िया", "accha": "अच्छा", "achha": "अच्छा",
        "acchi": "अच्छी", "acche": "अच्छे", "bekar": "बेकार", "ganda": "गंदा", "thik": "ठीक", "theek": "ठीक",
        "pakka": "पक्का", "sach": "सच", "jhooth": "झूठ", "tension": "टेंशन", "mat": "मत",
        "nahi": "नहीं", "nhi": "नहीं", "na": "ना", "haan": "हाँ", "kuch": "कुछ", "bilkul": "बिल्कुल",
        "toh": "तो", "to": "तो", "bhi": "भी", "hi": "ही", "aur": "और", "ya": "या", "ek": "एक", "do": "दो", "teen": "तीन",
        "lekin": "लेकिन", "magar": "मगर", "isliye": "इसलिए", "kyunki": "क्योंकि", "sirf": "सिर्फ",
        "agar": "अगर", "warna": "वरना", "jab": "जब", "tab": "तब", "thoda": "थोड़ा", "thodi": "थोड़ी", "thode": "थोड़े",
        "kamaal": "कमाल", "shukriya": "शुक्रिया", "dhanyavad": "धन्यवाद", "namaste": "नमस्ते"
    }

    # Comprehensive Colloquial Telugu Roman-to-Script Lexicon
    ROMAN_TO_TELUGU = {
        # Pronouns & Addressing
        "nenu": "నేను", "meeru": "మీరు", "nuvvu": "నువ్వు", "mee": "మీ", "naa": "నా",
        "ma": "మా", "maa": "మా", "neeku": "నీకు", "meeku": "మీకు", "naaku": "నాకు",
        "atanu": "అతను", "aame": "ఆమె", "vaaru": "వారు", "vallu": "వాళ్లు",
        "memu": "మేము", "manamu": "మనము", "manaki": "మనకి", "vaallu": "వాళ్లు",
        "anna": "అన్నా", "thammudu": "తమ్ముడు", "akka": "అక్క", "chelli": "చెల్లి",
        "amma": "అమ్మ", "nanna": "నాన్న", "mithrama": "మిత్రమా", "garu": "గారు",
        "sir": "సార్", "boss": "బాస్",

        # Interrogatives & Demonstratives
        "enti": "ఏంటి", "emi": "ఏమి", "enduku": "ఎందుకు", "ekkada": "ఎక్కడ",
        "eppudu": "ఎప్పుడు", "ela": "ఎలా", "evaru": "ఎవరు", "edi": "ఏది",
        "idi": "ఇది", "adi": "అది", "aa": "ఆ", "ee": "ఈ", "ikkada": "ఇక్కడ", "akkada": "అక్కడ",
        "anni": "అన్నీ", "enni": "ఎన్ని", "enta": "ఎంత",

        # Common Nouns & Adjectives
        "paniki": "పనికి", "pani": "పని", "peru": "పేరు", "illu": "ఇల్లు",
        "office": "ఆఫీస్", "samayam": "సమయం", "roju": "రోజు", "eeroju": "ఈరోజు",
        "repu": "రేపు", "ninna": "నిన్న", "chala": "చాలా", "bagundi": "బాగుంది",
        "bavundi": "బావుంది", "manchi": "మంచి", "chedda": "చెడ్డ", "pedda": "పెద్ద",
        "chinna": "చిన్న", "santosham": "సంతోషం", "kopam": "కోపం",
        "smell": "వాసన", "taste": "రుచి",

        # Verbs & Auxiliary
        "chesanu": "చేశాను", "chestara": "చేస్తారా", "cheyyandi": "చేయండి", "chey": "చేయి",
        "vastanu": "వస్తాను", "randi": "రండి", "vellandi": "వెళ్లండి", "velthanu": "వెళ్తాను",
        "chudandi": "చూడండి", "chusa": "చూశాను", "cheppandi": "చెప్పండి", "cheppu": "చెప్పు",
        "unnaru": "ఉన్నారు", "unnanu": "ఉన్నాను", "undi": "ఉంది", "ledu": "లేదు",
        "nachaledu": "నచ్చలేదు", "nachindi": "నచ్చింది",
        "kavali": "కావాలి", "vaddu": "వద్దు", "telusu": "తెలుసు", "teliyadu": "తెలియదు",

        # Conjunctions & Particles
        "mariyu": "మరియు", "kaani": "కానీ", "kabatti": "కాబట్టి", "kuda": "కూడా",
        "dhanyavadalu": "ధన్యవాదాలు", "namaskaram": "నమస్కారం", "avunu": "అవును", "kaadu": "కాదు"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("transliteration", {})
        self.enabled = config.get("pipeline", {}).get("enable_transliteration", True)
        self.transliterate_fn = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initializes indic_transliteration package if installed."""
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            self.sanscript = sanscript
            self.transliterate_fn = transliterate
        except ImportError:
            self.transliterate_fn = None

    def normalize(self, transcript: str, language_segments: List[Dict[str, str]], source_lang: Optional[str] = None) -> str:
        """
        Normalizes transcript by converting romanized words belonging to Indic languages
        into native script where appropriate.
        """
        if not self.enabled:
            return transcript

        normalized_tokens = []
        for seg in language_segments:
            word = seg.get("text", "")
            lang = seg.get("lang", "en")

            if lang in self.SCRIPT_MAP:
                mapped = self._fallback_indic_map(word, lang)
                normalized_tokens.append(mapped)
            else:
                normalized_tokens.append(word)

        return " ".join(normalized_tokens)

    COLLOQUIAL_TAMIL_TO_STANDARD = {
        # Negative verbs
        "புடிக்கல": "பிடிக்கவில்லை", "புடிக்கலை": "பிடிக்கவில்லை", "பிடிக்கல": "பிடிக்கவில்லை", "பிடிக்கலை": "பிடிக்கவில்லை",
        "தெரியல": "தெரியவில்லை", "தெரியலை": "தெரியவில்லை", "புரியல": "புரியவில்லை", "புரியலை": "புரியவில்லை",
        "வரல": "வரவில்லை", "வரலை": "வரவில்லை", "போகல": "போகவில்லை", "போகலை": "போகவில்லை",
        "பண்ணல": "செய்யவில்லை", "பண்ணலை": "செய்யவில்லை", "செய்யல": "செய்யவில்லை", "செய்யலை": "செய்யவில்லை",
        "சொல்லல": "சொல்லவில்லை", "சொல்லலை": "சொல்லவில்லை", "கேக்கல": "கேட்கவில்லை", "கேக்கலை": "கேட்கவில்லை",
        "கேட்கல": "கேட்கவில்லை", "கேட்கலை": "கேட்கவில்லை", "பாக்கல": "பார்க்கவில்லை", "பாக்கலை": "பார்க்கவில்லை",
        "பார்க்கல": "பார்க்கவில்லை", "பார்க்கலை": "பார்க்கவில்லை", "சாப்பிடல": "சாப்பிடவில்லை", "சாப்பிடலை": "சாப்பிடவில்லை",
        "தூங்கல": "தூங்கவில்லை", "தூங்கலை": "தூங்கவில்லை", "முடியல": "முடியவில்லை", "முடியலை": "முடியவில்லை",
        "கிடைக்கல": "கிடைக்கவில்லை", "கெடைக்கல": "கிடைக்கவில்லை",

        # Positive verbs & auxiliaries
        "புடிச்சிருக்கு": "பிடித்திருக்கிறது", "பிடிச்சிருக்கு": "பிடித்திருக்கிறது",
        "புடிச்சிருக்குது": "பிடித்திருக்கிறது", "பிடிச்சிருக்குது": "பிடித்திருக்கிறது",
        "புடிச்சிருந்தது": "பிடித்திருந்தது", "பிடிச்சிருந்தது": "பிடித்திருந்தது",
        "புடிச்சது": "பிடித்தது", "பிடிச்சது": "பிடித்தது",
        "பண்றேன்": "செய்கிறேன்", "பண்ணுறேன்": "செய்கிறேன்", "பண்ணுறான்": "செய்கிறான்", "பண்றான்": "செய்கிறான்",
        "பண்ணுறா": "செய்கிறாள்", "பண்றா": "செய்கிறாள்", "பண்ணுறாங்க": "செய்கிறார்கள்", "பண்றாங்க": "செய்கிறார்கள்",
        "பண்ணு": "செய்", "பண்ணிட்டேன்": "செய்துவிட்டேன்", "பண்ணலாம்": "செய்யலாம்",
        "போறேன்": "போகிறேன்", "போறான்": "போகிறான்", "போறா": "போகிறாள்", "போறாங்க": "போகிறார்கள்",
        "வர்றேன்": "வருகிறேன்", "வர்றான்": "வருகிறான்", "வர்றா": "வருகிறாள்", "வர்றாங்க": "வருகிறார்கள்",
        "வரேன்": "வருகிறேன்", "வராங்க": "வருகிறார்கள்", "சொல்றேன்": "சொல்கிறேன்", "சொல்றான்": "சொல்கிறான்",
        "சொல்றாங்க": "சொல்கிறார்கள்", "பாக்குறேன்": "பார்க்கிறேன்", "பாக்குறான்": "பார்க்கிறான்", "பாக்குறாங்க": "பார்க்கிறார்கள்",
        "கேக்குறேன்": "கேட்கிறேன்", "கேக்குறான்": "கேட்கிறான்", "கேக்குறாங்க": "கேட்கிறார்கள்",
        "இருக்கு": "இருக்கிறது", "இருக்குது": "இருக்கிறது", "இருக்குறேன்": "இருக்கிறேன்", "இருக்கேன்": "இருக்கிறேன்",
        "இருக்கான்": "இருக்கிறான்", "இருக்காங்க": "இருக்கிறார்கள்", "இருக்கோம்": "இருக்கிறோம்",
        "இல்ல": "இல்லை", "இல்லாம": "இல்லாமல்", "ரொம்ப": "மிகவும்", "நல்லா": "நன்றாக",
        "இப்போ": "இப்போது", "இப்ப": "இப்போது", "அப்போ": "அப்போது", "அப்ப": "அப்போது",
        "எப்போ": "எப்போது", "எப்ப": "எப்போது", "இங்க": "இங்கே", "அங்க": "அங்கே", "எங்க": "எங்கே",
        "எப்டி": "எப்படி", "மெதுவா": "மெதுவாக", "ஆச்சு": "ஆகிவிட்டது", "ஆயிடுச்சு": "ஆகிவிட்டது",
        "ஆகிடுச்சு": "ஆகிவிட்டது", "நேரமாச்சு": "நேரமாகிவிட்டது",
        "நாளைக்கு": "நாளை", "இன்னைக்கு": "இன்று", "நேத்துக்கு": "நேற்று",
        "வராரு": "வருகிறார்", "வர்றாரு": "வருகிறார்", "போறாரு": "போகிறார்", "சொல்றாரு": "சொல்கிறார்",
        "பண்ணனும்": "நாம் செய்ய வேண்டும்", "செய்யணும்": "நாம் செய்ய வேண்டும்", "சமைக்கணும்": "நாம் சமைக்க வேண்டும்",
        "பாக்கணும்": "பார்க்க வேண்டும்", "கேக்கணும்": "கேட்க வேண்டும்", "சாப்பிடணும்": "சாப்பிட வேண்டும்",
        "போகணும்": "போக வேண்டும்", "வரணும்": "வர வேண்டும்",
        "டின்னர்": "இரவு உணவு", "லஞ்ச்": "மதிய உணவு", "பிரேக்பாஸ்ட்": "காலை உணவு",
        "ஸ்மெல்": "வாசனை", "டேஸ்ட்": "சுவை", "கலர்": "நிறம்", "சவுண்ட்": "சத்தம்",
        "லுக்": "தோற்றம்", "டைம்": "நேரம்", "ப்ராப்ளம்": "பிரச்சனை", "ப்ராப்லம்": "பிரச்சனை", "ஐடியா": "யோசனை"
    }

    COLLOQUIAL_HINDI_TO_STANDARD = {
        "नही": "नहीं", "ह": "है", "कहा": "कहाँ", "यहा": "यहाँ", "वहा": "वहाँ",
        "स्मेल": "खुशबू", "टेस्ट": "स्वाद", "साउंड": "आवाज़", "टाइम": "समय",
        "प्रॉब्लम": "समस्या", "फास्ट": "तेज़", "स्लो": "धीमा", "वेरी": "बहुत"
    }

    def _fallback_indic_map(self, word: str, lang: str) -> str:
        """Dictionary-based mapping for colloquial romanized terms and spoken Indic forms."""
        w = word.lower().strip()

        if lang == "ta":
            # Check Roman Tanglish lexicon first
            if w in self.ROMAN_TO_TAMIL:
                return self.ROMAN_TO_TAMIL[w]
            # Check colloquial Tamil script normalization
            if word in self.COLLOQUIAL_TAMIL_TO_STANDARD:
                return self.COLLOQUIAL_TAMIL_TO_STANDARD[word]
            return word

        elif lang == "te":
            return getattr(self, "ROMAN_TO_TELUGU", {}).get(w, word)

        elif lang == "hi":
            # Check Roman Hinglish lexicon first
            if w in self.ROMAN_TO_DEVANAGARI:
                return self.ROMAN_TO_DEVANAGARI[w]
            # Check colloquial Hindi script normalization
            if word in self.COLLOQUIAL_HINDI_TO_STANDARD:
                return self.COLLOQUIAL_HINDI_TO_STANDARD[word]
            return word

        return word
