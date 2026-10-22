"""
Comprehensive Multilingual Basic Lexicon & Kinship Dictionary
Covers relations / kinship terms, greetings, daily vocabulary, food, time,
places, everyday verbs, numbers, and common questions across:
- Tamil (Script + Tanglish)
- Hindi (Devanagari + Hinglish)
- Telugu (Script + Telglish)
- Kannada (Script + Kanglish)
- Malayalam (Script + Manglish)
- English (Target & Source)
"""

from typing import Optional, Dict, Any
import re

# =========================================================================
# 1. COMPREHENSIVE KINSHIP / RELATIONSHIPS LEXICON
# =========================================================================

KINSHIP_CONCEPTS = {
    # Father
    "father": {
        "en": "father",
        "ta": ["அப்பா", "தந்தை", "தகப்பனார்"],
        "ta_roman": ["appa", "thadhai", "thanthai", "thagappanar"],
        "hi": ["पिताजी", "पिता", "पापा", "बापू", "अब्बू"],
        "hi_roman": ["pitaji", "pita", "papa", "bapu", "abbu"],
        "te": ["నాన్న", "తండ్రి", "పిత"],
        "te_roman": ["nanna", "tandri", "pitha"],
        "kn": ["ಅಪ್ಪ", "ತಂದೆ"],
        "kn_roman": ["appa", "thande"],
        "ml": ["അച്ഛൻ", "പിതാവ്"],
        "ml_roman": ["achan", "pithavu"]
    },
    # Mother
    "mother": {
        "en": "mother",
        "ta": ["அம்மா", "தாய்", "அன்னை"],
        "ta_roman": ["amma", "thaai", "thay", "annai"],
        "hi": ["माँ", "माताजी", "माता", "मम्मी", "अम्मी"],
        "hi_roman": ["maa", "mataji", "mata", "mummy", "ammi"],
        "te": ["అమ్మ", "తల్లి", "మాత"],
        "te_roman": ["amma", "thalli", "matha"],
        "kn": ["ಅಮ್ಮ", "ತಾಯಿ"],
        "kn_roman": ["amma", "thaayi"],
        "ml": ["അമ്മ", "മാതാവ്"],
        "ml_roman": ["amma", "mathavu"]
    },
    # Elder Brother
    "elder_brother": {
        "en": "elder brother",
        "ta": ["அண்ணா", "அண்ணன்", "மூத்த சகோதரன்"],
        "ta_roman": ["anna", "annan", "annaa", "mootha sagodharan"],
        "hi": ["भैया", "बड़े भाई", "भाई साहब", "अग्रज"],
        "hi_roman": ["bhaiya", "bade bhai", "bhai sahab", "bada bhai"],
        "te": ["అన్న", "అన్నయ్య", "పెద్దన్న"],
        "te_roman": ["anna", "annayya", "peddanna"],
        "kn": ["ಅಣ್ಣ", "ದೊಡ್ಡಣ್ಣ"],
        "kn_roman": ["anna", "doddanna"],
        "ml": ["ചേട്ടൻ", "ജ്യേഷ്ഠൻ"],
        "ml_roman": ["chettan", "jyeshtan"]
    },
    # Younger Brother
    "younger_brother": {
        "en": "younger brother",
        "ta": ["தம்பி", "இளைய சகோதரன்"],
        "ta_roman": ["thambi", "thamby", "ilaya sagodharan"],
        "hi": ["छोटा भाई", "अनुज"],
        "hi_roman": ["chhota bhai", "chota bhai", "anuj"],
        "te": ["తమ్ముడు"],
        "te_roman": ["thammudu", "tammudu"],
        "kn": ["ತಮ್ಮ"],
        "kn_roman": ["thamma"],
        "ml": ["അനിയൻ"],
        "ml_roman": ["aniyan"]
    },
    # Brother (General)
    "brother": {
        "en": "brother",
        "ta": ["சகோதரன்", "அண்ணன் தம்பி", "உடன்பிறப்பு"],
        "ta_roman": ["sagodharan", "bro"],
        "hi": ["भाई", "सहोदर"],
        "hi_roman": ["bhai", "sahodar", "bro"],
        "te": ["సోదరుడు"],
        "te_roman": ["sodarudu", "bro"],
        "kn": ["ಸಹೋದರ"],
        "kn_roman": ["sahodara"],
        "ml": ["സഹോദരൻ"],
        "ml_roman": ["sahodaran"]
    },
    # Elder Sister
    "elder_sister": {
        "en": "elder sister",
        "ta": ["அக்கா", "அக்காள்", "மூத்த சகோதரி"],
        "ta_roman": ["akka", "akkal", "akkaa"],
        "hi": ["दीदी", "बड़ी बहन", "जीजी"],
        "hi_roman": ["didi", "badi behen", "badi bahin", "jiji"],
        "te": ["అక్క", "అక్కయ్య"],
        "te_roman": ["akka", "akkayya"],
        "kn": ["ಅಕ್ಕ"],
        "kn_roman": ["akka"],
        "ml": ["ചേച്ചി"],
        "ml_roman": ["chechi"]
    },
    # Younger Sister
    "younger_sister": {
        "en": "younger sister",
        "ta": ["தங்கை", "தங்கச்சி", "இளைய சகோதரி"],
        "ta_roman": ["thangai", "thangachi", "thangachiye"],
        "hi": ["छोटी बहन"],
        "hi_roman": ["chhoti behen", "choti behen", "choti bahin"],
        "te": ["చెల్లి", "చెల్లెలు"],
        "te_roman": ["chelli", "chellelu"],
        "kn": ["ತಂಗಿ"],
        "kn_roman": ["thangi"],
        "ml": ["അനിയത്തി"],
        "ml_roman": ["aniyathi"]
    },
    # Sister (General)
    "sister": {
        "en": "sister",
        "ta": ["சகோதரி"],
        "ta_roman": ["sagodhari", "sistah"],
        "hi": ["बहन"],
        "hi_roman": ["behen", "bahin", "behna"],
        "te": ["సోదరి"],
        "te_roman": ["sodari"],
        "kn": ["ಸಹೋದರಿ"],
        "kn_roman": ["sahodari"],
        "ml": ["സഹോദരി"],
        "ml_roman": ["sahodari"]
    },
    # Paternal Uncle (Elder - Father's elder brother)
    "paternal_elder_uncle": {
        "en": "uncle (father's elder brother)",
        "ta": ["பெரியப்பா"],
        "ta_roman": ["periyappa", "periappa"],
        "hi": ["ताऊ", "ताऊजी", "बड़े पापा"],
        "hi_roman": ["tau", "tauji", "bade papa"],
        "te": ["పెద్దనాన్న"],
        "te_roman": ["peddananna", "pedda nanna"],
        "kn": ["ದೊಡ್ಡಪ್ಪ"],
        "kn_roman": ["doddappa"],
        "ml": ["വലിയച്ഛൻ"],
        "ml_roman": ["valiyachan"]
    },
    # Paternal Uncle (Younger - Father's younger brother)
    "paternal_younger_uncle": {
        "en": "uncle (father's younger brother)",
        "ta": ["சித்தப்பா", "சின்னப்பா"],
        "ta_roman": ["chithappa", "sithappa", "chittappa", "chinnappa"],
        "hi": ["चाचा", "चाचाजी", "काका"],
        "hi_roman": ["chacha", "chachaji", "kaka"],
        "te": ["బాబాయ్", "చిన్నాన్న"],
        "te_roman": ["babai", "baabay", "chinnanna"],
        "kn": ["ಚಿಕ್ಕಪ್ಪ"],
        "kn_roman": ["chikkappa"],
        "ml": ["ചിറ്റപ്പൻ", "ഇളയച്ഛൻ"],
        "ml_roman": ["chittappan", "ilayachan"]
    },
    # Maternal Uncle (Mother's brother)
    "maternal_uncle": {
        "en": "maternal uncle",
        "ta": ["மாமா", "தாய்மாமன்"],
        "ta_roman": ["mama", "maman", "thaimaman"],
        "hi": ["मामा", "मामाजी"],
        "hi_roman": ["mama", "mamaji"],
        "te": ["మామయ్య", "మేనమామ"],
        "te_roman": ["mamayya", "menamama", "mama"],
        "kn": ["ಮಾವ"],
        "kn_roman": ["maava", "mama"],
        "ml": ["അമ്മാവൻ"],
        "ml_roman": ["ammavan", "mama"]
    },
    # Paternal Aunt (Elder - Father's elder brother's wife / Mother's elder sister)
    "elder_aunt": {
        "en": "aunt (mother's elder sister)",
        "ta": ["பெரியம்மா"],
        "ta_roman": ["periyamma", "periamma"],
        "hi": ["ताई", "ताईजी", "बड़ी मम्मी"],
        "hi_roman": ["tai", "taiji", "badi mummy"],
        "te": ["పెద్దమ్మ"],
        "te_roman": ["peddamma"],
        "kn": ["ದೊಡ್ಡಮ್ಮ"],
        "kn_roman": ["doddamma"],
        "ml": ["വലിയമ്മ"],
        "ml_roman": ["valiyamma"]
    },
    # Younger Aunt (Father's younger brother's wife / Mother's younger sister)
    "younger_aunt": {
        "en": "aunt (mother's younger sister)",
        "ta": ["சித்தி", "சின்னம்மா"],
        "ta_roman": ["chithi", "sithi", "chinnamma"],
        "hi": ["चाची", "चाचीजी", "काकी", "मौसी"],
        "hi_roman": ["chachi", "chachiji", "kaki", "mausi", "mousi"],
        "te": ["పిన్ని", "చిన్నమ్మ"],
        "te_roman": ["pinni", "chinnamma"],
        "kn": ["ಚಿಕ್ಕಮ್ಮ"],
        "kn_roman": ["chikkamma"],
        "ml": ["ചിറ്റമ്മ", "ഇളയമ്മ"],
        "ml_roman": ["chittamma", "ilayamma"]
    },
    # Father's Sister / Mother-in-law
    "father_sister_aunt": {
        "en": "aunt (father's sister)",
        "ta": ["அத்தை", "மாமி"],
        "ta_roman": ["athai", "athaiye", "mami"],
        "hi": ["बुआ", "फूफी", "मामी"],
        "hi_roman": ["bua", "buaji", "foofi", "mami", "mamiji"],
        "te": ["అత్తయ్య", "మేనత్త", "అత్త"],
        "te_roman": ["atthayya", "menattha", "atta"],
        "kn": ["ಅತ್ತೆ"],
        "kn_roman": ["atthe"],
        "ml": ["അമ്മായി"],
        "ml_roman": ["ammayi"]
    },
    # Grandfather
    "grandfather": {
        "en": "grandfather",
        "ta": ["தாத்தா", "பாட்டனார்"],
        "ta_roman": ["thatha", "thaathaa", "paattanar"],
        "hi": ["दादा", "दादाजी", "नाना", "नानाजी"],
        "hi_roman": ["dada", "dadaji", "nana", "nanaji"],
        "te": ["తాత", "తాతయ్య"],
        "te_roman": ["thatha", "thathayya", "tata", "tatayya"],
        "kn": ["ಅಜ್ಜ"],
        "kn_roman": ["ajja"],
        "ml": ["മുത്തശ്ശൻ", "അപ്പൂപ്പൻ"],
        "ml_roman": ["muthashan", "appooppan"]
    },
    # Grandmother
    "grandmother": {
        "en": "grandmother",
        "ta": ["பாட்டி", "அம்மம்மா", "அப்பாத்தா"],
        "ta_roman": ["paatti", "patti", "ammamma", "appaatha"],
        "hi": ["दादी", "दादीजी", "नानी", "नानीजी"],
        "hi_roman": ["dadi", "dadiji", "nani", "naniji"],
        "te": ["నానమ్మ", "అమ్మమ్మ", "బామ్మ"],
        "te_roman": ["naanamma", "ammamma", "baamma"],
        "kn": ["ಅಜ್ಜಿ"],
        "kn_roman": ["ajji"],
        "ml": ["മുത്തശ്ശി", "അമ്മൂമ്മ"],
        "ml_roman": ["muthasshi", "ammoomma"]
    },
    # Son
    "son": {
        "en": "son",
        "ta": ["மகன்", "புதல்வன்"],
        "ta_roman": ["magan", "puthalvan"],
        "hi": ["बेटा", "पुत्र"],
        "hi_roman": ["beta", "putra"],
        "te": ["కొడుకు", "కుమారుడు"],
        "te_roman": ["koduku", "kumarudu"],
        "kn": ["ಮಗ"],
        "kn_roman": ["maga"],
        "ml": ["മകൻ"],
        "ml_roman": ["makan"]
    },
    # Daughter
    "daughter": {
        "en": "daughter",
        "ta": ["மகள்", "புதல்வி"],
        "ta_roman": ["magal", "puthalvi"],
        "hi": ["बेटी", "पुत्री"],
        "hi_roman": ["beti", "putri"],
        "te": ["కూతురు", "కుమార్తె"],
        "te_roman": ["kuthuru", "koothuru", "kumarthe"],
        "kn": ["ಮಗಳು"],
        "kn_roman": ["magalu"],
        "ml": ["മകൾ"],
        "ml_roman": ["makal"]
    },
    # Grandson
    "grandson": {
        "en": "grandson",
        "ta": ["பேரன்"],
        "ta_roman": ["peran"],
        "hi": ["पोता", "नाती"],
        "hi_roman": ["pota", "nati"],
        "te": ["మనవడు"],
        "te_roman": ["manavadu"],
        "kn": ["ಮೊಮ್ಮಗ"],
        "kn_roman": ["mommaga"],
        "ml": ["പേരക്കുട്ടി", "കൊച്ചുമകൻ"],
        "ml_roman": ["perakkutty", "kochumakan"]
    },
    # Granddaughter
    "granddaughter": {
        "en": "granddaughter",
        "ta": ["பேத்தி"],
        "ta_roman": ["pethi", "paethi"],
        "hi": ["पोती", "नातिन"],
        "hi_roman": ["poti", "natin"],
        "te": ["మనవరాలు"],
        "te_roman": ["manavaralu"],
        "kn": ["ಮೊಮ್ಮಗಳು"],
        "kn_roman": ["mommagalu"],
        "ml": ["കൊച്ചുമകൾ"],
        "ml_roman": ["kochumakal"]
    },
    # Husband
    "husband": {
        "en": "husband",
        "ta": ["கணவன்", "புருஷன்", "ஆத்துக்காரர்"],
        "ta_roman": ["kanavan", "purushan", "aathukkarar"],
        "hi": ["पति", "शौहर", "घरवाला"],
        "hi_roman": ["pati", "shauhar", "gharwala"],
        "te": ["భర్త", "మొగుడు"],
        "te_roman": ["bhartha", "mogudu"],
        "kn": ["ಗಂಡ"],
        "kn_roman": ["ganda"],
        "ml": ["ഭർത്താവ്"],
        "ml_roman": ["bharthavu"]
    },
    # Wife
    "wife": {
        "en": "wife",
        "ta": ["மனைவி", "பொண்டாட்டி", "சம்சாரம்"],
        "ta_roman": ["manaivi", "pondatti", "samsaram"],
        "hi": ["पत्नी", "बीवी", "घरवाली"],
        "hi_roman": ["patni", "biwi", "gharwali"],
        "te": ["భార్య", "పెళ్ళాం"],
        "te_roman": ["bharya", "pellam"],
        "kn": ["ಹೆಂಡತಿ"],
        "kn_roman": ["hendathi"],
        "ml": ["ഭാര്യ"],
        "ml_roman": ["bharya"]
    },
    # Friend
    "friend": {
        "en": "friend",
        "ta": ["நண்பன்", "நண்பா", "தோழன்", "தோழி", "மச்சி", "மச்சான்"],
        "ta_roman": ["nanban", "nanba", "thozhan", "thozhi", "machi", "machan"],
        "hi": ["दोस्त", "यार", "मित्र", "सखा", "सहेली"],
        "hi_roman": ["dost", "yaar", "mitra", "sakha", "saheli"],
        "te": ["స్నేహితుడు", "మిత్రుడు", "దోస్త్"],
        "te_roman": ["snehithudu", "mitrudu", "dost"],
        "kn": ["ಗೆಳೆಯ", "ಸ್ನೇಹಿತ"],
        "kn_roman": ["geleya", "snehitha"],
        "ml": ["കൂട്ടുകാരൻ", "സ്നേഹിതൻ"],
        "ml_roman": ["koottukaran", "snehithan"]
    },
    # Child / Baby
    "child": {
        "en": "child",
        "ta": ["குழந்தை", "பாப்பா", "பிள்ளை"],
        "ta_roman": ["kuzhandhai", "kulantha", "paappa", "pillai"],
        "hi": ["बच्चा", "बच्ची", "शिशु", "बालक"],
        "hi_roman": ["bacha", "bachcha", "bachi", "shishu"],
        "te": ["పిల్లలు", "పాప", "బాబు"],
        "te_roman": ["pillalu", "paapa", "baabu"],
        "kn": ["ಮಗು", "ಕೂಸು"],
        "kn_roman": ["magu", "koosu"],
        "ml": ["കുഞ്ഞ്", "കുട്ടി"],
        "ml_roman": ["kunju", "kutti"]
    }
}

# =========================================================================
# 2. DAY-TO-DAY BASIC WORDS & COMMON VOCABULARY
# =========================================================================

DAILY_VOCABULARY = {
    # Greetings
    "hello": {
        "en": "hello",
        "ta": ["வணக்கம்"], "ta_roman": ["vanakkam"],
        "hi": ["नमस्ते", "नमस्कार", "प्रणाम"], "hi_roman": ["namaste", "namaskar", "pranam"],
        "te": ["నమస్కారం", "నమస్తే"], "te_roman": ["namaskaram", "namaste"],
        "kn": ["ನಮಸ್ಕಾರ"], "kn_roman": ["namaskara"],
        "ml": ["നമസ്കാരം"], "ml_roman": ["namaskaram"]
    },
    "thank_you": {
        "en": "thank you",
        "ta": ["நன்றி", "ரொம்ப நன்றி"], "ta_roman": ["nandri", "romba nandri", "thanks"],
        "hi": ["धन्यवाद", "शुक्रिया", "बहुत धन्यवाद"], "hi_roman": ["dhanyavad", "shukriya", "bahut dhanyavad"],
        "te": ["ధన్యవాదాలు", "కృతజ్ఞతలు"], "te_roman": ["dhanyavadalu", "kruthagnathalu"],
        "kn": ["ಧನ್ಯವಾದಗಳು"], "kn_roman": ["dhanyavadagalu"],
        "ml": ["നന്ദി"], "ml_roman": ["nandi"]
    },
    "welcome": {
        "en": "welcome",
        "ta": ["வரவேற்கிறோம்", "வாங்க"], "ta_roman": ["varaverkirom", "vaanga"],
        "hi": ["स्वागत है", "आपका स्वागत है"], "hi_roman": ["swagat hai", "aapka swagat hai"],
        "te": ["స్వాగతం"], "te_roman": ["swagatham"],
        "kn": ["ಸ್ವಾಗತ"], "kn_roman": ["swagata"],
        "ml": ["സ്വാഗതം"], "ml_roman": ["swagatham"]
    },
    "please": {
        "en": "please",
        "ta": ["தயவுசெய்து"], "ta_roman": ["dayavuseithu", "please"],
        "hi": ["कृपया", "मेहरबानी करके"], "hi_roman": ["kripya", "meherbani karke", "please"],
        "te": ["దయచేసి"], "te_roman": ["dayachesi", "please"],
        "kn": ["ದಯವಿಟ್ಟು"], "kn_roman": ["dayavittu"],
        "ml": ["ദയവായി"], "ml_roman": ["dayavayi"]
    },
    "sorry": {
        "en": "sorry",
        "ta": ["மன்னிக்கவும்", "மன்னிச்சிடுங்க"], "ta_roman": ["mannikkavum", "mannichudunga", "sorry"],
        "hi": ["माफ़ कीजिए", "क्षमा करें"], "hi_roman": ["maaf kijiye", "kshama karein", "sorry"],
        "te": ["క్షమించండి"], "te_roman": ["kshaminchandi", "sorry"],
        "kn": ["ಕ್ಷಮಿಸಿ"], "kn_roman": ["kshamisi"],
        "ml": ["ക്ഷമിക്കണം"], "ml_roman": ["kshamikanam"]
    },

    # Time / Days
    "today": {
        "en": "today",
        "ta": ["இன்று", "இன்னைக்கு"], "ta_roman": ["indru", "innaiku", "innaikku"],
        "hi": ["आज"], "hi_roman": ["aaj"],
        "te": ["ఈరోజు", "నేడు"], "te_roman": ["eeroju", "nedu"],
        "kn": ["ಇಂದು", "ಇವತ್ತು"], "kn_roman": ["indu", "ivattu"],
        "ml": ["ഇന്ന്"], "ml_roman": ["innu"]
    },
    "tomorrow": {
        "en": "tomorrow",
        "ta": ["நாளை", "நாளைக்கு"], "ta_roman": ["naalai", "naalaiku", "nalaiku"],
        "hi": ["कल"], "hi_roman": ["kal"],
        "te": ["రేపు"], "te_roman": ["repu"],
        "kn": ["ನಾಳೆ"], "kn_roman": ["naale"],
        "ml": ["നാളെ"], "ml_roman": ["naale"]
    },
    "yesterday": {
        "en": "yesterday",
        "ta": ["நேற்று", "நேத்து"], "ta_roman": ["netru", "nethu"],
        "hi": ["कल", "बीता हुआ कल"], "hi_roman": ["kal", "beeta kal"],
        "te": ["నిన్న"], "te_roman": ["ninna"],
        "kn": ["ನಿನ್ನೆ"], "kn_roman": ["ninne"],
        "ml": ["ഇന്നലെ"], "ml_roman": ["innale"]
    },
    "now": {
        "en": "now",
        "ta": ["இப்போது", "இப்ப", "இப்போ"], "ta_roman": ["ippothu", "ippo", "ippa"],
        "hi": ["अब", "अभी"], "hi_roman": ["ab", "abhi"],
        "te": ["ఇప్పుడు"], "te_roman": ["ippudu"],
        "kn": ["ಈಗ"], "kn_roman": ["eega"],
        "ml": ["ഇപ്പോൾ"], "ml_roman": ["ippol"]
    },
    "morning": {
        "en": "morning",
        "ta": ["காலை"], "ta_roman": ["kaalai", "kalai"],
        "hi": ["सुबह", "प्रातः"], "hi_roman": ["subah", "pratah"],
        "te": ["ఉదయం"], "te_roman": ["udayam"],
        "kn": ["ಬೆಳಗ್ಗೆ"], "kn_roman": ["belagge"],
        "ml": ["രാവിലെ"], "ml_roman": ["ravile"]
    },
    "night": {
        "en": "night",
        "ta": ["இரவு", "ராத்திரி"], "ta_roman": ["iravu", "raathiri", "raatri"],
        "hi": ["रात", "रात्रि"], "hi_roman": ["raat", "ratri"],
        "te": ["రాత్రి"], "te_roman": ["ratri"],
        "kn": ["ರಾತ್ರಿ"], "kn_roman": ["raatri"],
        "ml": ["രാത്രി"], "ml_roman": ["rathri"]
    },

    # Food & Drink
    "water": {
        "en": "water",
        "ta": ["தண்ணீர்", "தண்ணி", "நீர்"], "ta_roman": ["thanneer", "thanni", "neer"],
        "hi": ["पानी", "जल"], "hi_roman": ["paani", "pani", "jal"],
        "te": ["నీరు", "నీళ్లు", "మంచినీళ్లు"], "te_roman": ["neeru", "neellu", "manchineellu"],
        "kn": ["ನೀರು"], "kn_roman": ["neeru"],
        "ml": ["വെള്ളം", "ജലം"], "ml_roman": ["vellam", "jalam"]
    },
    "food": {
        "en": "food",
        "ta": ["உணவு", "சாப்பாடு", "சோறு"], "ta_roman": ["unavu", "saappadu", "saapadu", "soru"],
        "hi": ["खाना", "भोजन"], "hi_roman": ["khana", "bhojan"],
        "te": ["ఆహారం", "కూడు", "భోజనం"], "te_roman": ["aahaaram", "bhojanam"],
        "kn": ["ಊಟ", "ಆಹಾರ"], "kn_roman": ["oota", "aahara"],
        "ml": ["ഭക്ഷണം", "ഊണ്"], "ml_roman": ["bhakshanam", "oonu"]
    },
    "tea": {
        "en": "tea",
        "ta": ["தேநீர்", "டீ"], "ta_roman": ["theaneer", "tea"],
        "hi": ["चाय"], "hi_roman": ["chai", "chay"],
        "te": ["టీ", "తేనీరు"], "te_roman": ["tea", "theeneeru"],
        "kn": ["ಚಹಾ", "ಟೀ"], "kn_roman": ["chaha", "tea"],
        "ml": ["ചായ"], "ml_roman": ["chaya"]
    },
    "coffee": {
        "en": "coffee",
        "ta": ["காபி", "குளம்பி"], "ta_roman": ["coffee", "kaapi"],
        "hi": ["कॉफ़ी"], "hi_roman": ["coffee", "kafi"],
        "te": ["కాఫీ"], "te_roman": ["coffee", "kaafi"],
        "kn": ["ಕಾಫಿ"], "kn_roman": ["coffee"],
        "ml": ["കാപ്പി"], "ml_roman": ["kappi"]
    },
    "milk": {
        "en": "milk",
        "ta": ["பால்"], "ta_roman": ["paal"],
        "hi": ["दूध"], "hi_roman": ["doodh", "dudh"],
        "te": ["పాలు"], "te_roman": ["paalu"],
        "kn": ["ಹಾಲು"], "kn_roman": ["haalu"],
        "ml": ["പാൽ"], "ml_roman": ["paal"]
    },

    # House & Places
    "house": {
        "en": "house",
        "ta": ["வீடு", "இல்லம்"], "ta_roman": ["veedu", "veetla", "illam"],
        "hi": ["घर", "मकान"], "hi_roman": ["ghar", "makan"],
        "te": ["ఇల్లు", "గృహం"], "te_roman": ["illu", "gruham"],
        "kn": ["ಮನೆ"], "kn_roman": ["mane"],
        "ml": ["വീട്"], "ml_roman": ["veedu"]
    },
    "school": {
        "en": "school",
        "ta": ["பள்ளி", "பள்ளிக்கூடம்"], "ta_roman": ["palli", "pallikoodam"],
        "hi": ["स्कूल", "विद्यालय", "पाठशाला"], "hi_roman": ["school", "vidyalaya"],
        "te": ["పాఠశాల", "బడి"], "te_roman": ["paatasaala", "badi"],
        "kn": ["ಶಾಲೆ"], "kn_roman": ["shaale"],
        "ml": ["സ്കൂൾ", "പള്ളിക്കൂടം"], "ml_roman": ["school", "pallikoodam"]
    },
    "work": {
        "en": "work",
        "ta": ["வேலை"], "ta_roman": ["velai", "vela"],
        "hi": ["काम", "कार्य"], "hi_roman": ["kaam", "karya"],
        "te": ["పని"], "te_roman": ["pani"],
        "kn": ["ಕೆಲಸ"], "kn_roman": ["kelasa"],
        "ml": ["ജോലി", "പണി"], "ml_roman": ["joli", "pani"]
    },
    "money": {
        "en": "money",
        "ta": ["பணம்", "காசு"], "ta_roman": ["panam", "kaasu", "kasu"],
        "hi": ["पैसा", "धन", "रुपया"], "hi_roman": ["paisa", "paise", "dhan", "rupya"],
        "te": ["డబ్బు", "ధనం"], "te_roman": ["dabbu", "dhanam"],
        "kn": ["ಹಣ", "ದುಡ್ಡು"], "kn_roman": ["hana", "duddu"],
        "ml": ["പണം", "കാശു"], "ml_roman": ["panam", "kaasu"]
    },

    # Basic Responses
    "yes": {
        "en": "yes",
        "ta": ["ஆம்", "ஆமா", "ஆமாம்", "சரி"], "ta_roman": ["aama", "aamam", "aam", "sari", "seri"],
        "hi": ["हाँ", "जी हाँ", "सही"], "hi_roman": ["haan", "ji haan", "sahi"],
        "te": ["అవును", "సరే"], "te_roman": ["avunu", "sare"],
        "kn": ["ಹೌದು", "ಸರಿ"], "kn_roman": ["haudu", "sari"],
        "ml": ["അതെ", "ശരി"], "ml_roman": ["athe", "shari"]
    },
    "no": {
        "en": "no",
        "ta": ["இல்லை", "இல்ல"], "ta_roman": ["illai", "illa"],
        "hi": ["नहीं", "जी नहीं"], "hi_roman": ["nahi", "nahin", "ji nahi"],
        "te": ["కాదు", "లేదు"], "te_roman": ["kaadu", "ledu"],
        "kn": ["ಇಲ್ಲ", "ಅಲ್ಲ"], "kn_roman": ["illa", "alla"],
        "ml": ["ഇല്ല", "അല്ല"], "ml_roman": ["illa", "alla"]
    },

    # Common Questions
    "what": {
        "en": "what",
        "ta": ["என்ன"], "ta_roman": ["enna"],
        "hi": ["क्या"], "hi_roman": ["kya"],
        "te": ["ఏమిటి", "ఏంటి"], "te_roman": ["emiti", "enti"],
        "kn": ["ಏನು"], "kn_roman": ["enu"],
        "ml": ["എന്ത്"], "ml_roman": ["enthu"]
    },
    "where": {
        "en": "where",
        "ta": ["எங்கே", "எங்க"], "ta_roman": ["engae", "enga", "enge"],
        "hi": ["कहाँ", "किधर"], "hi_roman": ["kahan", "kidhar"],
        "te": ["ఎక్కడ"], "te_roman": ["ekkada"],
        "kn": ["ಎಲ್ಲಿ"], "kn_roman": ["elli"],
        "ml": ["എവിടെ"], "ml_roman": ["evide"]
    },
    "when": {
        "en": "when",
        "ta": ["எப்போது", "எப்போ"], "ta_roman": ["eppothu", "eppo"],
        "hi": ["कब"], "hi_roman": ["kab"],
        "te": ["ఎప్పుడు"], "te_roman": ["eppudu"],
        "kn": ["ಯಾವಾಗ"], "kn_roman": ["yaavaaga"],
        "ml": ["എപ്പോൾ"], "ml_roman": ["eppol"]
    },
    "why": {
        "en": "why",
        "ta": ["ஏன்", "எதற்கு"], "ta_roman": ["yean", "en", "edhuku"],
        "hi": ["क्यों"], "hi_roman": ["kyun", "kyu"],
        "te": ["ఎందుకు"], "te_roman": ["enduku"],
        "kn": ["ಏಕೆ"], "kn_roman": ["yake"],
        "ml": ["എന്തുകൊണ്ട്"], "ml_roman": ["enthukondu"]
    },
    "how": {
        "en": "how",
        "ta": ["எப்படி", "எப்டி"], "ta_roman": ["eppadi", "epdi"],
        "hi": ["कैसे", "कैसा"], "hi_roman": ["kaise", "kaisa"],
        "te": ["ఎలా"], "te_roman": ["ela"],
        "kn": ["ಹೇಗೆ"], "kn_roman": ["hege"],
        "ml": ["എങ്ങനെ"], "ml_roman": ["engane"]
    },
    "who": {
        "en": "who",
        "ta": ["யார்"], "ta_roman": ["yaaru", "yaar"],
        "hi": ["कौन"], "hi_roman": ["kaun"],
        "te": ["ఎవరు"], "te_roman": ["evaru"],
        "kn": ["ಯಾರು"], "kn_roman": ["yaaru"],
        "ml": ["ആര്"], "ml_roman": ["aaru"]
    }
}

# Precompute fast lookup index
# Map every known surface token -> concept dictionary
FAST_LOOKUP_INDEX = {}

def _build_index():
    all_catalogs = [KINSHIP_CONCEPTS, DAILY_VOCABULARY]
    for cat in all_catalogs:
        for concept_key, entry in cat.items():
            en_val = entry["en"]
            # English keys
            FAST_LOOKUP_INDEX[en_val.lower().strip()] = entry
            
            # Target language representations
            for lang_key, forms in entry.items():
                if lang_key == "en":
                    continue
                for f in forms:
                    cleaned_key = f.lower().strip()
                    if cleaned_key:
                        FAST_LOOKUP_INDEX[cleaned_key] = entry

_build_index()


def lookup_lexicon(query: str, source_lang: Optional[str] = None, target_lang: str = "en") -> Optional[str]:
    """
    Direct lexical lookup for kinship terms, basic day-to-day vocabulary, possessives, and demonstratives.
    Returns authoritative, context-accurate translation immediately without neural hallucinations.
    """
    if not query:
        return None

    clean_q = re.sub(r"[^\w\s\u0900-\u0DFF'-]", "", query).strip().lower()
    if not clean_q:
        return None

    tgt = (target_lang or "en").lower().strip()
    is_tgt_en = tgt in ["en", "eng_latn", "english"]

    # 1. Direct single-term lookup
    entry = FAST_LOOKUP_INDEX.get(clean_q)
    if entry:
        if is_tgt_en:
            res = entry["en"]
            return res[0].upper() + res[1:] if len(res) > 1 else res
        else:
            lang_map = {
                "ta": "ta", "tamil": "ta", "ta-en": "ta", "tanglish": "ta",
                "hi": "hi", "hindi": "hi", "hi-en": "hi", "hinglish": "hi",
                "te": "te", "telugu": "te", "te-en": "te", "telglish": "te",
                "kn": "kn", "kannada": "kn", "ml": "ml", "malayalam": "ml"
            }
            tgt_code = lang_map.get(tgt, tgt)
            if tgt_code in entry and entry[tgt_code]:
                return entry[tgt_code][0]

    # 2. Possessive phrase parsing ("என் அண்ணா" -> "My elder brother")
    possessive_patterns = [
        # Tamil possessives
        (r"^(என்|என்னுடைய|என்னோட|எனது)\s+(.+)$", "My "),
        (r"^(எங்கள்|எங்களோட|நமது|நம்ம)\s+(.+)$", "Our "),
        (r"^(உன்|உன்னுடைய|உன்னோட|உமது)\s+(.+)$", "Your "),
        (r"^(அவரது|அவருடைய|அவனோட|அவன்)\s+(.+)$", "His "),
        (r"^(அவளது|அவளுடைய|அவளோட|அவள்)\s+(.+)$", "Her "),
        # Tanglish Roman
        (r"^(en|ennoda|ennudaiya|enadhu)\s+(.+)$", "My "),
        (r"^(enga|namma|namakku)\s+(.+)$", "Our "),
        (r"^(un|unnoda|unnudaiya)\s+(.+)$", "Your "),
        # Hindi possessives
        (r"^(मेरा|मेरी|मेरे)\s+(.+)$", "My "),
        (r"^(हमारा|हमारी|हमारे)\s+(.+)$", "Our "),
        (r"^(तेरा|तेरी|तेरे|तुम्हारा|तुम्हारी|तुम्हारे|आपका|आपकी|आपके)\s+(.+)$", "Your "),
        # Hinglish Roman
        (r"^(mera|meri|mere)\s+(.+)$", "My "),
        (r"^(hamara|humara|hamari|humari|hamare|humare)\s+(.+)$", "Our "),
        (r"^(tera|teri|tere|tumhara|tumhari|tumhare|aapka|aapki|aapke)\s+(.+)$", "Your "),
        # Telugu possessives
        (r"^(నా|నా యొక్క)\s+(.+)$", "My "),
        (r"^(మా|మా యొక్క|మన)\s+(.+)$", "Our "),
        (r"^(నీ|మీ)\s+(.+)$", "Your "),
        # Telglish Roman
        (r"^(naa|na)\s+(.+)$", "My "),
        (r"^(maa|ma|mana)\s+(.+)$", "Our "),
        (r"^(nee|mee)\s+(.+)$", "Your ")
    ]

    for pat, prefix_en in possessive_patterns:
        m = re.match(pat, clean_q)
        if m:
            core_term = m.group(2).strip()
            core_entry = FAST_LOOKUP_INDEX.get(core_term)
            if core_entry and is_tgt_en:
                return prefix_en + core_entry["en"].lower()

    # 3. Demonstrative Sentence Patterns ("அவர் என் அண்ணா" -> "He is my elder brother.")
    demonstrative_patterns = [
        # Tamil
        (r"^(அவர்|இவர்|அவன்|இவன்)\s+(என்|என்னுடைய|என்னோட)\s+(.+)$", "He is my "),
        (r"^(அவள்|இவள்)\s+(என்|என்னுடைய|என்னோட)\s+(.+)$", "She is my "),
        (r"^(இது|அது)\s+(என்|என்னுடைய|என்னோட)\s+(.+)$", "This is my "),
        # Tanglish
        (r"^(avar|ivar|avan|ivan)\s+(en|ennoda)\s+(.+)$", "He is my "),
        (r"^(aval|ival)\s+(en|ennoda)\s+(.+)$", "She is my "),
        (r"^(idhu|adhu)\s+(en|ennoda)\s+(.+)$", "This is my "),
        # Hindi
        (r"^(वह|ये|वो)\s+(मेरे|मेरा)\s+(.+?)(?:\s+हैं|\s+है)?$", "He is my "),
        (r"^(वह|ये|वो)\s+मेरी\s+(.+?)(?:\s+हैं|\s+है)?$", "She is my "),
        (r"^(यह|ये)\s+(मेरा|मेरी|मेरे)\s+(.+?)(?:\s+हैं|\s+है)?$", "This is my "),
        # Hinglish
        (r"^(woh|wo|ye|yeh)\s+(mere|mera)\s+(.+?)(?:\s+hain|\s+hai)?$", "He is my "),
        (r"^(woh|wo|ye|yeh)\s+meri\s+(.+?)(?:\s+hain|\s+hai)?$", "She is my "),
        # Telugu
        (r"^(ఆయన|ఈయన|వారు|వీరు|అతను)\s+నా\s+(.+)$", "He is my "),
        (r"^(ఆమె|ఈమె)\s+నా\s+(.+)$", "She is my ")
    ]

    for pat, prefix_en in demonstrative_patterns:
        m = re.match(pat, clean_q)
        if m:
            core_term = m.group(m.lastindex).strip()
            core_entry = FAST_LOOKUP_INDEX.get(core_term)
            if core_entry and is_tgt_en:
                return prefix_en + core_entry["en"].lower() + "."

    return None


def correct_kinship_hallucinations(source_text: str, translated_text: str) -> str:
    """
    Guards sentence-level translations against NLLB's known kinship hallucination traps.
    e.g. 'அவர் என் அண்ணா' -> NLLB outputs 'He's my grandmother' -> corrected to 'He is my elder brother'.
    """
    if not source_text or not translated_text:
        return translated_text

    s_clean = source_text.strip().lower()
    out = translated_text.strip()

    # Hallucination 1: அண்ணா / அண்ணன் / भैया / అన్న translated as grandmother
    if any(k in s_clean for k in ["அண்ணா", "அண்ணன்", "bhaiya", "bhaye", "భయ్యా", "అన్న", "అన్నయ్య"]):
        if re.search(r"\bgrandmother\b", out, re.IGNORECASE):
            out = re.sub(r"\bgrandmother\b", "elder brother", out, flags=re.IGNORECASE)
            out = re.sub(r"\bmy grandmother\b", "my elder brother", out, flags=re.IGNORECASE)
        if re.search(r"\bgrand mother\b", out, re.IGNORECASE):
            out = re.sub(r"\bgrand mother\b", "elder brother", out, flags=re.IGNORECASE)

    # Hallucination 2: பெரியப்பா / பெரியம்மா / சித்தப்பா / चाचा translated as grandfather / grandpa
    if any(k in s_clean for k in ["பெரியப்பா", "periyappa", "சித்தப்பா", "chithappa", "चाचा", "chacha", "బాబాయ్", "పెద్దనాన్న"]):
        if re.search(r"\b(grandfather|grandpa|grand-pa)\b", out, re.IGNORECASE):
            out = re.sub(r"\b(grandfather|grandpa|grand-pa)\b", "uncle", out, flags=re.IGNORECASE)
        if re.search(r"\bgrand father\b", out, re.IGNORECASE):
            out = re.sub(r"\bgrand father\b", "uncle", out, flags=re.IGNORECASE)
        if out.lower().startswith("grandpa "):
            out = "Uncle " + out[8:]
        if out.lower().startswith("grandfather "):
            out = "Uncle " + out[12:]

    # Hallucination 3: அக்கா translated as 'AKA' or 'AKA.'
    if any(k in s_clean for k in ["அக்கா", "அக்காள்", "akkaa"]):
        if re.search(r"\bAKA\b", out):
            out = re.sub(r"\bAKA\b", "elder sister", out)

    # Capitalize leading character
    if out and len(out) > 1:
        out = out[0].upper() + out[1:]

    return out
