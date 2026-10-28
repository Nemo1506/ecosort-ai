"""Intelligent Waste Sorting Assistant Knowledge Base & Query Engine.

Provides authoritative municipal answers for tricky, ambiguous items that
regular vision classifiers or consumers frequently encounter, complete with
conversational AI assistant response generation tailored to global jurisdictions.
"""
import re
from typing import Any, Dict, List, Optional

STOP_WORDS = {
    "can", "i", "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "be", "do", "does", "or", "and", "my", "how", "what", "where",
    "which", "should", "it", "this", "that", "marked", "symbol", "recycle",
    "recycling", "recycled", "put", "throw", "dispose", "disposal", "bin",
    "trash", "waste", "go", "into", "regular", "placed", "item", "items"
}

ECO_ASSISTANT_FAQ: List[Dict[str, Any]] = [
    {
        "keywords": ["black", "plastic", "tray", "food", "symbol", "microwave", "cpet", "meal"],
        "question": "Can I recycle black plastic food trays marked with a recycling symbol?",
        "summary": "🍱 Black Plastic Trays: Almost universally non-recyclable in curbside bins.",
        "answer": (
            "**No. Despite displaying the recycling resin triangle (e.g. #1 PET or #5 PP), black plastic food trays cannot be recycled in standard curbside single-stream bins.**\n\n"
            "**Why:** Sorting facilities use Near-Infrared (NIR) optical sensors to scan conveyor belts. Carbon black pigment absorbs the infrared light beam entirely instead of reflecting it back, rendering the tray invisible to optical sorters. As a result, it gets sorted into residual landfill trash or contaminates clear plastic bales.\n\n"
            "**Action Steps:**\n"
            "1. Scrape out and rinse any food residue.\n"
            "2. Check if your city specifically uses new detectable NIR black pigment scanners (e.g. DetectaBlack); otherwise:\n"
            "3. Place in **General Landfill Trash / Non-Recyclable Bin**.\n"
            "4. Transparent or clear plastic lids can be separated and recycled if accepted locally."
        ),
        "stream": "General Trash / Landfill",
        "badge_color": "#1e293b",
    },
    {
        "keywords": ["pizza", "grease", "box", "oil"],
        "question": "Can I recycle a pizza box that has grease on the bottom?",
        "summary": "🍕 Pizza Boxes: Only clean cardboard is recyclable.",
        "answer": (
            "**No, greasy cardboard cannot be recycled.** The grease and oil cannot be separated from paper fibers during "
            "the water-based pulping process, ruining entire batches of recycled paper.\n\n"
            "**Action Steps:**\n"
            "1. Tear the box in half.\n"
            "2. Put the clean top lid in the **Cardboard / Recycling Bin**.\n"
            "3. Put the greasy, cheese-stained bottom in your **Compost / Organics Bin** (if accepted) or **General Trash**."
        ),
        "stream": "Cardboard / Compost Split",
        "badge_color": "#2563eb",
    },
    {
        "keywords": ["coffee", "cup", "lid", "starbucks", "takeaway"],
        "question": "Are single-use paper coffee cups recyclable?",
        "summary": "☕ Disposable Coffee Cups: Most cups are NOT recyclable in paper bins.",
        "answer": (
            "**Standard paper coffee cups are coated with a hidden polyethylene plastic lining** to prevent leaking, "
            "making them non-recyclable in standard paper streams.\n\n"
            "**Action Steps:**\n"
            "1. Remove the plastic lid (#5 or #6): Rinse and place in **Plastics Recycling** if your local city accepts lids.\n"
            "2. Remove the cardboard sleeve: Place in **Paper / Cardboard Recycling**.\n"
            "3. The cup body: Place in **General Landfill Trash** unless your municipality specifically has poly-coated carton processing."
        ),
        "stream": "Mixed Streams",
        "badge_color": "#4b5563",
    },
    {
        "keywords": ["bubble", "wrap", "film", "bag", "grocery", "plastic", "shipping"],
        "question": "Can bubble wrap and plastic shopping bags go in the home recycling bin?",
        "summary": "🫧 Plastic Film & Bubble Wrap: Severe mechanical tangler — store drop-off only.",
        "answer": (
            "**No, never put flexible film or bubble wrap in curbside recycling bins.** They wrap around rotating sorting disks at recovery facilities, shutting down entire plant operations.\n\n"
            "**Action Steps:**\n"
            "1. Pop bubbles to reduce volume or reuse for future packages.\n"
            "2. Bundle clean plastic bags and bubble wrap together.\n"
            "3. Drop off at grocery store collection bins (front-of-store plastic film drop-off points)."
        ),
        "stream": "Supermarket Drop-off",
        "badge_color": "#0284c7",
    },
    {
        "keywords": ["pod", "capsule", "nespresso", "keurig", "k-cup", "coffee"],
        "question": "Are aluminium or plastic coffee pods recyclable?",
        "summary": "☕ Coffee Pods: Too small for standard sorters — use manufacturer take-back.",
        "answer": (
            "**Standard curbside bins cannot process single-serve coffee pods.** They are too small for mechanical screens (they fall through glass crushing grates) and contain wet organic coffee grounds trapped inside.\n\n"
            "**Action Steps:**\n"
            "1. Use manufacturer mail-back bags (Nespresso recycling bags or TerraCycle drop-offs).\n"
            "2. If cleaning manually: scoop out coffee grounds into Compost, rinse the foil/cup, and recycle only if your municipality accepts items under 2 inches.\n"
            "3. Otherwise, dispose of in **General Trash**."
        ),
        "stream": "Specialized Mail-Back / Trash",
        "badge_color": "#b45309",
    },
    {
        "keywords": ["chip", "crisp", "packet", "wrapper", "candy", "snack", "mylar", "foil"],
        "question": "Can I recycle chip bags, crisp packets, or candy wrappers?",
        "summary": "🥔 Chip Bags / Crisp Packets: Metallized plastic composite — Landfill Trash.",
        "answer": (
            "**No. Snack wrappers are made from metallized plastic film (BOPP + vaporized aluminium)** bonded together, which cannot be separated mechanically.\n\n"
            "**Action Steps:**\n"
            "1. Do the 'scrunch test': If you scrunch the packet and it springs back flat, it is metallized plastic.\n"
            "2. Check for supermarket flexible plastic collection or TerraCycle snack wrapper points.\n"
            "3. If no specialized drop-off is available, place in **General Landfill Trash**."
        ),
        "stream": "General Trash / TerraCycle",
        "badge_color": "#4b5563",
    },
    {
        "keywords": ["battery", "swollen", "lithium", "aa", "aaa", "fire"],
        "question": "How do I safely dispose of swollen or used batteries?",
        "summary": "🔋 Battery Disposal: Severe fire hazard — never in regular trash.",
        "answer": (
            "**Batteries are domestic hazardous e-waste.** Lithium-ion and alkaline batteries can ignite garbage trucks "
            "and leach lead/cadmium into groundwater.\n\n"
            "**Action Steps:**\n"
            "1. Put clear tape over the positive (+) and negative (-) terminals of lithium and 9V batteries to prevent short circuits.\n"
            "2. Place swollen batteries in a container filled with sand or kitty litter.\n"
            "3. Drop off at retail collection points (Best Buy, Home Depot, supermarket battery bins) or city HHW stations."
        ),
        "stream": "Hazardous E-Waste",
        "badge_color": "#dc2626",
    },
    {
        "keywords": ["styrofoam", "polystyrene", "foam", "peanuts", "packaging"],
        "question": "Is Styrofoam (Expanded Polystyrene #6) recyclable?",
        "summary": "📦 Styrofoam: Almost universally excluded from curbside recycling.",
        "answer": (
            "**Polystyrene is 95% air and easily breaks into microplastic beads**, causing severe contamination at sorting facilities.\n\n"
            "**Action Steps:**\n"
            "1. Packing peanuts: Re-use them for mailing or donate them to local shipping stores (UPS / FedEx).\n"
            "2. Rigid foam blocks: Look for specialized EPS drop-off depots.\n"
            "3. If no EPS drop-off exists locally, place in **General Landfill Trash**."
        ),
        "stream": "General Trash / Specialized Drop-off",
        "badge_color": "#4b5563",
    },
    {
        "keywords": ["mirror", "broken", "glass", "window", "pyrex", "ceramic", "mug"],
        "question": "Can I put broken mirrors, ceramic mugs, or Pyrex in the glass recycling bin?",
        "summary": "🪞 Treated Glass & Ceramics: NOT recyclable with beverage bottles.",
        "answer": (
            "**NO.** Mirrors, window panes, Pyrex cookware, and ceramics have different chemical compositions and higher melting "
            "temperatures than container glass. Mixing even a small ceramic shard into a glass furnace ruins the entire melt.\n\n"
            "**Action Steps:**\n"
            "1. Wrap the broken pieces securely in thick newspaper or bubble wrap to protect sanitation workers from cuts.\n"
            "2. Label the package 'Broken Glass' and place in **General Landfill Trash**."
        ),
        "stream": "General Trash (Safety Wrapped)",
        "badge_color": "#dc2626",
    },
    {
        "keywords": ["blister", "pack", "medicine", "pill", "tablet"],
        "question": "Are medicine blister packs (foil + plastic) recyclable?",
        "summary": "💊 Medicine Blister Packs: Bonded composite packaging.",
        "answer": (
            "**Blister packs are made of aluminium foil bonded tightly to PVC plastic**, making mechanical separation difficult in standard curbside sorting.\n\n"
            "**Action Steps:**\n"
            "1. Check if local pharmacies offer specialized blister pack recycling collection (e.g. TerraCycle schemes).\n"
            "2. If no pharmacy take-back is available, place in **General Trash**.\n"
            "3. Cardboard medicine packaging boxes go to **Paper / Cardboard Recycling**."
        ),
        "stream": "Pharmacy Drop-off or Trash",
        "badge_color": "#4b5563",
    },
    {
        "keywords": ["toothpaste", "tube", "lotion", "cosmetic", "squeeze"],
        "question": "Are toothpaste tubes and cosmetic squeeze tubes recyclable?",
        "summary": "🪥 Toothpaste Tubes: Multi-layer laminate with heavy paste residue.",
        "answer": (
            "**Most squeeze tubes are multi-layer laminates (aluminium sandwiched between plastic)** and contain stubborn residue that contaminates recycling.\n\n"
            "**Action Steps:**\n"
            "1. Check for the HDPE #2 recyclable logo (some newer Colgate/Crest tubes are pure HDPE).\n"
            "2. Cut off the top, rinse out remaining paste completely.\n"
            "3. If not marked with pure HDPE #2 symbol, place in **General Trash** or pharmacy cosmetic take-back bins."
        ),
        "stream": "General Trash / Specialty Take-back",
        "badge_color": "#4b5563",
    },
    {
        "keywords": ["tetra", "pak", "milk", "carton", "juice"],
        "question": "Are Tetra Paks and aseptic beverage cartons recyclable?",
        "summary": "🧃 Tetra Paks: Multi-layer composite (Paper + Polyethylene + Aluminium).",
        "answer": (
            "**Yes in many modern cities, but acceptance depends on local facilities.** They are processed with hydrapulpers to recover virgin paper fiber.\n\n"
            "**Action Steps:**\n"
            "1. Empty all liquid residue completely and rinse lightly.\n"
            "2. Flatten the carton to conserve collection volume.\n"
            "3. Leave plastic screw caps attached (modern optical sorters capture caps).\n"
            "4. Check local city rules: usually placed in **Mixed Recycling / Yellow Bin**."
        ),
        "stream": "Carton Recycling / Mixed Stream",
        "badge_color": "#16a34a",
    },
    {
        "keywords": ["receipt", "thermal", "paper", "bpa"],
        "question": "Can cash register receipts be recycled with paper?",
        "summary": "🧾 Thermal Receipts: Contain BPA / BPS chemicals — do NOT recycle.",
        "answer": (
            "**Most shiny checkout receipts are printed on thermal paper coated with Bisphenol A (BPA) or BPS.**\n\n"
            "Recycling them introduces toxic endocrine disruptors into recycled paper products like toilet paper and paper towels.\n\n"
            "**Action Steps:**\n"
            "1. Do a scratch test: If scratching with a coin or fingernail leaves a dark mark, it is thermal paper.\n"
            "2. Place thermal receipts into **General Landfill Trash**."
        ),
        "stream": "General Trash",
        "badge_color": "#4b5563",
    },
    {
        "keywords": ["aerosol", "spray", "can", "deodorant", "paint"],
        "question": "Can aerosol spray cans be placed in metal recycling?",
        "summary": "🧴 Aerosol Cans: Only when 100% completely empty.",
        "answer": (
            "**Aerosol cans are pressurized steel or aluminium.** If they still contain propellant, they can explode in the compactor.\n\n"
            "**Action Steps:**\n"
            "1. Depress the nozzle until no hissing sound or spray comes out.\n"
            "2. If completely empty, place in **Metal Recyclables (Blue/Yellow bin)**.\n"
            "3. If partially full (especially paint, pesticides, or chemicals), route to **Household Hazardous Waste (HHW)**."
        ),
        "stream": "Metal (Empty) or Hazardous (Partial)",
        "badge_color": "#0284c7",
    },
    {
        "keywords": ["lightbulb", "bulb", "cfl", "led", "fluorescent", "tube"],
        "question": "Where should I dispose of lightbulbs and fluorescent tubes?",
        "summary": "💡 Lightbulbs: CFLs contain toxic mercury vapour.",
        "answer": (
            "**Never put fluorescent tubes or compact fluorescents (CFLs) into regular trash or glass bins.** They contain hazardous mercury gas.\n\n"
            "**Action Steps:**\n"
            "1. CFLs & fluorescent tubes: Take to home improvement stores (IKEA, Home Depot, B&Q) or municipal e-waste depots.\n"
            "2. LED bulbs: Recyclable at electronic waste drop-offs.\n"
            "3. Incandescent / Halogen bulbs: Place in **General Landfill Trash** wrapped safely."
        ),
        "stream": "Hazardous E-Waste Drop-off",
        "badge_color": "#dc2626",
    },
    {
        "keywords": ["foil", "aluminium", "baking", "tray"],
        "question": "Can aluminium foil and baking trays be recycled?",
        "summary": "🥫 Aluminium Foil: 100% infinitely recyclable if clean.",
        "answer": (
            "**Aluminium foil is pure metal and saves 95% of energy compared to mining virgin bauxite.**\n\n"
            "**Action Steps:**\n"
            "1. Wipe or rinse off food crumbs and grease.\n"
            "2. Scrunch clean foil sheets together into a ball at least the size of a tennis ball so sorting machinery can detect it.\n"
            "3. Place in the **Metal / Recyclables Bin**."
        ),
        "stream": "Metal Recycling",
        "badge_color": "#0284c7",
    },
    {
        "keywords": ["bioplastic", "pla", "compostable", "biodegradable", "fork", "cup"],
        "question": "Can I put PLA compostable plastic into the regular plastic recycling bin?",
        "summary": "🌱 Compostable Plastics (PLA): Never mix with standard plastics.",
        "answer": (
            "**DO NOT put PLA into plastic recycling.** PLA (Polylactic Acid) melts at a much lower temperature than PET or HDPE, ruining an entire batch of recycled plastic flakes.\n\n"
            "**Action Steps:**\n"
            "1. Commercial compost bin: If certified 'OK Compost Industrial', place in industrial organics collection.\n"
            "2. Home composting: Most PLA will not degrade in backyard compost piles.\n"
            "3. If no industrial composting facility exists, place in **General Trash**."
        ),
        "stream": "Industrial Compost or Trash",
        "badge_color": "#78350f",
    },
]


def search_assistant_faq(query: str, min_score: int = 3) -> List[Dict[str, Any]]:
    """Search the FAQ database with stop word filtering and score thresholding."""
    if not query or not query.strip():
        return ECO_ASSISTANT_FAQ[:4]

    q_lower = query.lower()
    raw_words = re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower)
    content_words = [w for w in raw_words if w not in STOP_WORDS and len(w) > 2]

    scored = []
    for item in ECO_ASSISTANT_FAQ:
        score = 0
        # Check phrase and keyword matches
        for kw in item["keywords"]:
            if kw in q_lower:
                score += 4
        for word in content_words:
            if word in item["question"].lower():
                score += 3
            if word in item["answer"].lower():
                score += 1
        if score >= min_score:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top_score = scored[0][0]
        threshold = max(min_score, int(top_score * 0.5))
        return [item for s, item in scored if s >= threshold][:3]
    return []


def _resolve_jurisdiction_bin(stream_type: str, jurisdiction: str) -> Dict[str, str]:
    """Map conceptual waste stream to jurisdiction-specific bin name and color."""
    jur_key = jurisdiction.lower()
    if "united states" in jur_key or "us" in jur_key:
        mapping = {
            "trash": {"name": "Black / Gray Cart (Landfill Trash)", "color": "#475569"},
            "recycle": {"name": "Blue Cart (Curbside Single-Stream Recyclables)", "color": "#2563eb"},
            "compost": {"name": "Green Cart (Organics & Compost)", "color": "#15803d"},
            "hazardous": {"name": "Red / Designated HHW Depot (Hazardous Drop-off)", "color": "#dc2626"},
            "dropoff": {"name": "Supermarket / Retail Store Drop-off Point", "color": "#0284c7"},
        }
    elif "european" in jur_key or "eu" in jur_key:
        mapping = {
            "trash": {"name": "Restmüll / Grey Bin (Non-Recyclable Residual)", "color": "#475569"},
            "recycle": {"name": "Gelber Sack / Yellow Bin (Packaging & Dual System)", "color": "#eab308"},
            "compost": {"name": "Biotonne / Brown Organics Bin", "color": "#78350f"},
            "hazardous": {"name": "Wertstoffhof / Hazardous Recycling Center", "color": "#dc2626"},
            "dropoff": {"name": "Supermarket Reverse Vending / Store Container", "color": "#0284c7"},
        }
    elif "kingdom" in jur_key or "uk" in jur_key:
        mapping = {
            "trash": {"name": "General Waste Wheelie Bin (Landfill / Energy Recovery)", "color": "#475569"},
            "recycle": {"name": "Dry Mixed Recycling Wheelie Bin", "color": "#2563eb"},
            "compost": {"name": "Food Caddy / Brown Garden Waste Bin", "color": "#15803d"},
            "hazardous": {"name": "HWRC Household Waste Recycling Depot", "color": "#dc2626"},
            "dropoff": {"name": "Supermarket Front-of-Store Collection", "color": "#0284c7"},
        }
    elif "japan" in jur_key:
        mapping = {
            "trash": {"name": "Moeru Gomi (Burnable Trash / Designated City Bag)", "color": "#dc2626"},
            "recycle": {"name": "Pura (Plastic Packaging Resource) / Kan (Cans)", "color": "#2563eb"},
            "compost": {"name": "Nama Gomi (Raw Kitchen Food Scraps)", "color": "#15803d"},
            "hazardous": {"name": "Yugai Gomi (Hazardous City Collection Station)", "color": "#d97706"},
            "dropoff": {"name": "Station / Supermarket Resource Collection Box", "color": "#0284c7"},
        }
    elif "india" in jur_key:
        mapping = {
            "trash": {"name": "Black Bin (Domestic Sanitary & Non-Recyclable Waste)", "color": "#1e293b"},
            "recycle": {"name": "Blue Bin (Dry Recyclables / Sookha Kachra)", "color": "#2563eb"},
            "compost": {"name": "Green Bin (Wet Organics / Gila Kachra)", "color": "#15803d"},
            "hazardous": {"name": "Red Bin (Domestic Hazardous & E-Waste)", "color": "#dc2626"},
            "dropoff": {"name": "Kabadiwala / Authorized E-Waste Center", "color": "#0284c7"},
        }
    else:
        mapping = {
            "trash": {"name": "General Landfill Trash Bin", "color": "#475569"},
            "recycle": {"name": "Mixed Recycling Bin", "color": "#2563eb"},
            "compost": {"name": "Organics / Compost Bin", "color": "#15803d"},
            "hazardous": {"name": "Specialized Hazardous Waste Facility", "color": "#dc2626"},
            "dropoff": {"name": "Retail / Store Take-Back Depot", "color": "#0284c7"},
        }
    return mapping.get(stream_type, mapping["trash"])


def generate_assistant_response(query: str, jurisdiction: str = "United States") -> Dict[str, Any]:
    """Synthesize an authoritative, structured, and conversational AI assistant answer.

    Evaluates user input against municipal packaging regulations, optical sorting mechanics,
    and material science to provide direct verdicts, detailed scientific rationales, and
    jurisdiction-accurate disposal instructions.
    """
    if not query or not query.strip():
        bin_info = _resolve_jurisdiction_bin("recycle", jurisdiction)
        return {
            "headline": "EcoSort Waste Intelligence Advisor",
            "verdict": "💬 READY FOR INQUIRY",
            "verdict_badge": "#38bdf8",
            "verdict_icon": "💡",
            "direct_answer": "Ask any question about tricky consumer packaging, composite materials, or local disposal regulations.",
            "scientific_reason": "EcoSort AI cross-references Material Recovery Facility (MRF) optical sorting limits, municipal bin rules, and Life-Cycle Assessment data.",
            "action_steps": [
                "Type an item name (e.g. 'black plastic food tray', 'pizza box', 'bubble wrap').",
                "Review the direct verdict and local bin allocation for your active jurisdiction.",
                "Follow the preparation steps to prevent recycling stream contamination."
            ],
            "target_bin": bin_info["name"],
            "bin_color": bin_info["color"],
            "audio_summary": "EcoSort AI is ready. Ask any waste segregation or recycling inquiry.",
            "related_faq": ECO_ASSISTANT_FAQ[:3],
        }

    q = query.lower()

    # 1. Black Plastic Food Trays / Containers
    if ("black" in q and ("plastic" in q or "tray" in q or "container" in q or "food" in q or "cpet" in q)) or ("tray" in q and "black" in q):
        bin_info = _resolve_jurisdiction_bin("trash", jurisdiction)
        return {
            "headline": "Black Plastic Food Trays & Containers",
            "verdict": "❌ DO NOT RECYCLE IN CURBSIDE BINS (PLACE IN GENERAL TRASH)",
            "verdict_badge": "#dc2626",
            "verdict_icon": "❌",
            "direct_answer": (
                "**No, you cannot recycle black plastic food trays in regular curbside recycling bins**, "
                "even if they are stamped with a recycling symbol (such as #1 PET or #5 PP)."
            ),
            "scientific_reason": (
                "**The Optical Sorting Blind Spot:** Modern Material Recovery Facilities (MRFs) rely on "
                "Near-Infrared (NIR) optical sorting lasers to detect polymer types as items travel on conveyor belts. "
                "The carbon black pigment used in black trays completely absorbs the infrared light beam rather than reflecting "
                "it back. Because no light bounces back to the scanner, automated ejectors cannot identify the plastic type, "
                "causing it to either be discarded into residual landfill trash or contaminate clear plastic recycling bales."
            ),
            "action_steps": [
                "Scrape and rinse out all leftover food residue, gravy, or grease.",
                "Remove and discard any plastic film wrapping or absorbent meat pads into General Trash.",
                "Place the black plastic food tray into your designated **General Trash / Residual Landfill Cart**.",
                "If the package included a **clear, transparent plastic lid**, that lid *can* usually be recycled in your Plastics Bin."
            ],
            "target_bin": bin_info["name"],
            "bin_color": bin_info["color"],
            "audio_summary": "Black plastic food trays cannot be recycled curbside because infrared sorting lasers cannot detect black pigment. Place in general trash.",
            "related_faq": search_assistant_faq(query),
        }

    # 2. Pizza Boxes & Greasy Cardboard
    if "pizza" in q or ("grease" in q and ("box" in q or "cardboard" in q)):
        bin_compost = _resolve_jurisdiction_bin("compost", jurisdiction)
        return {
            "headline": "Pizza Boxes & Greasy Cardboard",
            "verdict": "⚠️ PARTIAL SPLIT REQUIRED (CLEAN LID = RECYCLE • GREASY BASE = COMPOST/TRASH)",
            "verdict_badge": "#f59e0b",
            "verdict_icon": "⚠️",
            "direct_answer": (
                "**Greasy pizza box bottoms cannot be recycled into paper.** Food oils and melted cheese cannot be separated "
                "from paper fibers during the water-based pulping process, ruining entire batches of recycled cardboard."
            ),
            "scientific_reason": (
                "In paper recycling, fibers are mixed with water in large hydrapulpers. Oil and water do not mix, and oil bonds "
                "to paper fibers, causing weak spots and grease holes in newly manufactured recycled cardboard."
            ),
            "action_steps": [
                "Tear the pizza box along the hinge.",
                "Place the clean, grease-free **top lid** into the **Recycling Bin**.",
                "Place the cheese-stained **greasy bottom** into your **Compost / Organics Bin** (if accepted) or **General Trash**.",
                "Throw away the plastic center pizza saver tripod into General Trash."
            ],
            "target_bin": f"{bin_compost['name']} (Greasy half)",
            "bin_color": bin_compost["color"],
            "audio_summary": "Tear the pizza box in half. Put the clean lid in recycling and the greasy bottom into compost or general trash.",
            "related_faq": search_assistant_faq(query),
        }

    # 3. Disposable Paper Coffee Cups
    if "coffee" in q and ("cup" in q or "lid" in q or "starbucks" in q or "takeaway" in q or "disposable" in q):
        bin_trash = _resolve_jurisdiction_bin("trash", jurisdiction)
        return {
            "headline": "Disposable Paper Coffee Cups",
            "verdict": "❌ NOT RECYCLABLE IN STANDARD PAPER BINS (GENERAL TRASH)",
            "verdict_badge": "#dc2626",
            "verdict_icon": "❌",
            "direct_answer": (
                "**Standard takeaway paper coffee cups cannot go into regular paper recycling.** "
                "They are laminated with a hidden polyethylene plastic liner that regular paper mills cannot separate."
            ),
            "scientific_reason": (
                "To prevent hot liquid from dissolving the cardboard, cups feature a tightly bonded inner plastic film. "
                "Unless your municipality specifically partners with specialized poly-coated carton pulping facilities, "
                "they act as contaminants in paper recycling streams."
            ),
            "action_steps": [
                "Pull off the plastic lid: Rinse and check if marked #5 PP (recycle if local program accepts small rigid lids).",
                "Slide off the cardboard sleeve: Place in your **Paper / Cardboard Recycling Bin**.",
                "Place the paper cup body into **General Landfill Trash**."
            ],
            "target_bin": bin_trash["name"],
            "bin_color": bin_trash["color"],
            "audio_summary": "Disposable paper coffee cups have a plastic lining and belong in general trash. Recycle the cardboard sleeve separately.",
            "related_faq": search_assistant_faq(query),
        }

    # 4. Bubble Wrap & Plastic Bags / Films
    if "bubble" in q or "film" in q or ("plastic" in q and ("bag" in q or "wrap" in q or "grocery" in q or "cling" in q)):
        bin_drop = _resolve_jurisdiction_bin("dropoff", jurisdiction)
        return {
            "headline": "Bubble Wrap & Flexible Plastic Film",
            "verdict": "⚠️ STORE DROP-OFF ONLY (NEVER IN CURBSIDE BINS)",
            "verdict_badge": "#f59e0b",
            "verdict_icon": "⚠️",
            "direct_answer": (
                "**Do NOT put bubble wrap or plastic bags into your curbside recycling cart.** "
                "They are notorious mechanical tanglers that shut down sorting machinery."
            ),
            "scientific_reason": (
                "Flexible low-density polyethylene (LDPE #4) wraps tightly around high-speed rotating axle screens and sorters "
                "at recycling facilities. Workers must manually shut down the facility and cut them out with utility knives."
            ),
            "action_steps": [
                "Reuse bubble wrap for storage, mailing, or cushioning fragile items.",
                "Bundle clean, dry plastic grocery bags, bubble wrap, and mailer sleeves together.",
                "Drop them off at participating supermarket front-of-store plastic film collection bins.",
                "If no store drop-off is accessible, discard in **General Trash**."
            ],
            "target_bin": bin_drop["name"],
            "bin_color": bin_drop["color"],
            "audio_summary": "Never put bubble wrap or plastic bags in curbside bins as they tangle machinery. Take them to supermarket drop-offs.",
            "related_faq": search_assistant_faq(query),
        }

    # 5. Batteries & Lithium Cells
    if "battery" in q or "batteries" in q or "lithium" in q or "aa" in q or "aaa" in q:
        bin_haz = _resolve_jurisdiction_bin("hazardous", jurisdiction)
        return {
            "headline": "Batteries & Rechargeable Lithium Cells",
            "verdict": "🚨 HAZARDOUS E-WASTE ONLY (CRITICAL FIRE RISK IN TRASH)",
            "verdict_badge": "#dc2626",
            "verdict_icon": "🚨",
            "direct_answer": (
                "**NEVER put batteries in regular trash or curbside recycling.** "
                "Lithium and alkaline batteries cause violent fires in waste collection trucks and sorting facilities."
            ),
            "scientific_reason": (
                "When compacted in trucks, lithium-ion batteries rupture, causing thermal runaway reactions reaching 1,000°C. "
                "Heavy metals like cadmium, nickel, and lead also leach into soil and groundwater if landfilled."
            ),
            "action_steps": [
                "Tape the positive (+) and negative (-) terminals with clear tape to prevent short circuits.",
                "If swollen or leaking, store in an open metal bucket with dry sand or cat litter.",
                "Take to retail battery collection points (Home Depot, Best Buy, IKEA, supermarkets) or municipal HHW centers."
            ],
            "target_bin": bin_haz["name"],
            "bin_color": bin_haz["color"],
            "audio_summary": "Never put batteries in curbside bins. They cause truck and depot fires. Tape terminals and drop off at battery collection sites.",
            "related_faq": search_assistant_faq(query),
        }

    # 6. Coffee Pods & Capsules
    if "pod" in q or "capsule" in q or "nespresso" in q or "keurig" in q or "k-cup" in q:
        bin_drop = _resolve_jurisdiction_bin("dropoff", jurisdiction)
        return {
            "headline": "Coffee Pods & Capsules (Aluminium / Plastic)",
            "verdict": "⚠️ SPECIALIZED TAKE-BACK PROGRAM (TOO SMALL FOR CURBSIDE BINS)",
            "verdict_badge": "#f59e0b",
            "verdict_icon": "⚠️",
            "direct_answer": (
                "**Standard curbside recycling bins cannot process single-serve coffee capsules.** "
                "They are too small for sorting screens and contain wet organic grounds."
            ),
            "scientific_reason": (
                "Automated sorting screens discard items smaller than 2-3 inches directly into landfill chutes. "
                "Wet coffee grounds also create moisture contamination in paper recycling."
            ),
            "action_steps": [
                "Use manufacturer mail-back bags (e.g. Nespresso prepaid recycling bags or Podback in UK).",
                "If using reusable pods, empty spent coffee grounds directly into food compost.",
                "If non-recyclable plastic pods, dispose of in **General Trash**."
            ],
            "target_bin": f"{bin_drop['name']} / Mail-Back",
            "bin_color": bin_drop["color"],
            "audio_summary": "Coffee pods are too small for curbside sorters. Use manufacturer mail-back bags or empty grounds into compost.",
            "related_faq": search_assistant_faq(query),
        }

    # 7. Broken Mirrors, Window Panes, Ceramics & Pyrex
    if "mirror" in q or "window" in q or "pyrex" in q or "ceramic" in q or "mug" in q or ("broken" in q and "glass" in q):
        bin_trash = _resolve_jurisdiction_bin("trash", jurisdiction)
        return {
            "headline": "Treated Glass, Ceramics & Broken Mirrors",
            "verdict": "❌ NEVER MIX WITH CONTAINER GLASS (GENERAL TRASH - WRAP SAFELY)",
            "verdict_badge": "#dc2626",
            "verdict_icon": "❌",
            "direct_answer": (
                "**Do NOT put mirrors, window glass, Pyrex, or ceramics in glass recycling bins.** "
                "They are not container glass and will ruin an entire batch of recycled glass bottles."
            ),
            "scientific_reason": (
                "Cookware glass, mirrors, and ceramics are formulated with borosilicates and chemical coatings giving them "
                "much higher melting points than soda-lime container glass. A single ceramic speck causes structural defects and bottle explosions."
            ),
            "action_steps": [
                "Wrap broken shards in several layers of newspaper or cardboard to protect sanitation workers.",
                "Tape the package securely and write 'Broken Glass' on the outside.",
                "Place into your **General Landfill Trash Bin**."
            ],
            "target_bin": bin_trash["name"],
            "bin_color": bin_trash["color"],
            "audio_summary": "Mirrors, ceramics, and Pyrex have different melting temperatures than bottle glass. Wrap safely and place in general trash.",
            "related_faq": search_assistant_faq(query),
        }

    # 8. Aluminum Foil & Baking Trays
    if "foil" in q or "aluminium" in q or "aluminum" in q or "baking tray" in q:
        bin_rec = _resolve_jurisdiction_bin("recycle", jurisdiction)
        return {
            "headline": "Aluminium Foil & Clean Food Trays",
            "verdict": "✅ 100% INFINITELY RECYCLABLE IF CLEAN (BALL UP TO TENNIS BALL SIZE)",
            "verdict_badge": "#16a34a",
            "verdict_icon": "✅",
            "direct_answer": (
                "**Yes! Clean aluminium foil and baking trays are 100% recyclable.** "
                "Recycling aluminium saves 95% of the energy needed to smelt new metal from raw ore."
            ),
            "scientific_reason": (
                "Aluminium does not degrade during recycling. However, small flat pieces of foil blow away or get misrouted "
                "into paper streams by optical sorting blowers. Balling them up allows eddy current sorters to repel them properly."
            ),
            "action_steps": [
                "Wipe or rinse off grease and baked-on food particles.",
                "Scrunch clean foil sheets together into a dense ball (at least 2-3 inches / tennis ball size).",
                "Place in your **Metal / Recyclables Cart**."
            ],
            "target_bin": bin_rec["name"],
            "bin_color": bin_rec["color"],
            "audio_summary": "Clean aluminium foil is infinitely recyclable. Wipe off food residue and ball it up into a tennis ball size before recycling.",
            "related_faq": search_assistant_faq(query),
        }

    # 9. Styrofoam / Polystyrene (EPS #6)
    if "styrofoam" in q or "polystyrene" in q or "foam" in q or "eps" in q:
        bin_trash = _resolve_jurisdiction_bin("trash", jurisdiction)
        return {
            "headline": "Styrofoam & Expanded Polystyrene (#6 EPS)",
            "verdict": "❌ NOT ACCEPTED IN CURBSIDE RECYCLING (GENERAL TRASH OR DEPOT)",
            "verdict_badge": "#dc2626",
            "verdict_icon": "❌",
            "direct_answer": (
                "**Styrofoam cups, takeout clamshells, and packaging foam cannot go into curbside bins.** "
                "It breaks into microplastic beads and is 95% air, making curbside collection economically unviable."
            ),
            "scientific_reason": (
                "EPS is extremely low-density. In standard recycling trucks, it shatters into thousands of tiny static beads that "
                "stick to paper and cardboard, rendering entire bales unmarketable."
            ),
            "action_steps": [
                "Takeout food containers: Scrape food and place in **General Trash**.",
                "Loose packing peanuts: Donate to local shipping stores (UPS / FedEx) for immediate reuse.",
                "Large rigid foam blocks: Check if your regional drop-off depot has an EPS thermal densifier."
            ],
            "target_bin": bin_trash["name"],
            "bin_color": bin_trash["color"],
            "audio_summary": "Styrofoam cannot be recycled in curbside carts because it shatters into static microplastics. Place in general trash or take to foam depots.",
            "related_faq": search_assistant_faq(query),
        }

    # 10. Thermal Receipts (BPA Paper)
    if "receipt" in q or "thermal" in q or "bpa" in q:
        bin_trash = _resolve_jurisdiction_bin("trash", jurisdiction)
        return {
            "headline": "Thermal Store Checkout Receipts",
            "verdict": "❌ DO NOT RECYCLE WITH PAPER (CONTAINS BPA/BPS ENDOCRINE DISRUPTORS)",
            "verdict_badge": "#dc2626",
            "verdict_icon": "❌",
            "direct_answer": (
                "**Do NOT put glossy checkout receipts into paper recycling.** "
                "They are coated with toxic Bisphenol A (BPA) or BPS chemical color developers."
            ),
            "scientific_reason": (
                "When thermal paper is recycled, BPA enters water systems and transfers into consumer paper goods like "
                "tissue paper, kitchen towels, and napkins. It is a proven endocrine disruptor."
            ),
            "action_steps": [
                "Scratch test: Scratch the receipt with a fingernail or coin; if a dark grey streak appears, it is thermal paper.",
                "Discard all thermal checkout slips into **General Landfill Trash**.",
                "Opt for digital or email receipts at checkout whenever possible."
            ],
            "target_bin": bin_trash["name"],
            "bin_color": bin_trash["color"],
            "audio_summary": "Thermal receipts contain toxic BPA and must never be recycled with paper. Place receipts in general trash.",
            "related_faq": search_assistant_faq(query),
        }

    # 11. Generic / Intelligent Synthesis Fallback
    matched_faq = search_assistant_faq(query, min_score=2)
    bin_rec = _resolve_jurisdiction_bin("recycle", jurisdiction)

    # Detect if query implies clean recyclables
    is_likely_recyclable = any(
        term in q for term in ["clean paper", "cardboard box", "aluminum can", "soda can", "tin can", "water bottle", "milk jug", "shampoo bottle", "glass bottle", "glass jar"]
    )

    if is_likely_recyclable:
        return {
            "headline": f"Recycling Guidance for '{query.title()}'",
            "verdict": "✅ ACCEPTED IN RECYCLABLES (EMPTY, CLEAN & DRY)",
            "verdict_badge": "#16a34a",
            "verdict_icon": "✅",
            "direct_answer": (
                f"**Yes! Standard clean packaging items like '{query}' are widely accepted in curbside recycling.** "
                "Follow standard preparation rules to ensure maximum material purity."
            ),
            "scientific_reason": (
                "Recyclable rigid plastics (#1 PET, #2 HDPE), cardboard, and aluminum have robust global circular markets. "
                "Keeping materials dry and grease-free maintains commercial recovery efficiency."
            ),
            "action_steps": [
                "Empty all liquid and scrape food residue completely.",
                "Rinse lightly; clean containers prevent mold and odor.",
                "Replace plastic screw caps tightly on bottles.",
                "Place in your **Recyclables Cart**."
            ],
            "target_bin": bin_rec["name"],
            "bin_color": bin_rec["color"],
            "audio_summary": f"{query} is accepted in your local recyclables bin. Empty liquids, rinse, and place in recycling.",
            "related_faq": matched_faq,
        }

    bin_trash = _resolve_jurisdiction_bin("trash", jurisdiction)
    return {
        "headline": f"Disposal Assessment: '{query.title()}'",
        "verdict": "🔍 COMPOSITE / SPECIAL HANDLING ADVISORY",
        "verdict_badge": "#38bdf8",
        "verdict_icon": "💡",
        "direct_answer": (
            f"For ambiguous items like **'{query}'**, always verify whether the product is a pure mono-material or a multi-layer composite. "
            "Items with heavy food contamination, mixed bonded layers, or pigments should default to General Trash to prevent contaminating clean recyclables."
        ),
        "scientific_reason": (
            "Recycling contamination costs municipal facilities millions of dollars annually in rejected bales and mechanical downtime. "
            "The universal industry golden rule is: **'When in doubt, find out — or leave it out (throw it in the trash).'**"
        ),
        "action_steps": [
            "Check the product packaging for resin codes (#1 to #7) or localized recycling logos.",
            "Separate components: discard soft liners and caps into their respective bins.",
            f"If unsure whether your regional plant accepts it, route to **{bin_trash['name']}** to prevent contamination.",
            "Consult the related authoritative guidelines below for detailed chemical breakdown."
        ],
        "target_bin": bin_trash["name"],
        "bin_color": bin_trash["color"],
        "audio_summary": f"For {query}, check for composite materials. When in doubt, place in general trash to prevent recycling contamination.",
        "related_faq": matched_faq,
    }
