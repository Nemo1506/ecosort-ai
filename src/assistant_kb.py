"""Intelligent Waste Sorting Assistant Knowledge Base & Query Engine.

Provides authoritative municipal answers for tricky, ambiguous items that
regular vision classifiers or consumers frequently encounter.
"""
from typing import Dict, List

ECO_ASSISTANT_FAQ = [
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


def search_assistant_faq(query: str) -> List[Dict]:
    """Search the FAQ database by keyword match or full text."""
    if not query or not query.strip():
        return ECO_ASSISTANT_FAQ[:5]

    q_lower = query.lower()
    q_words = set(q_lower.replace("?", "").replace(",", "").split())

    scored = []
    for item in ECO_ASSISTANT_FAQ:
        score = 0
        for kw in item["keywords"]:
            if kw in q_lower:
                score += 3
        for word in q_words:
            if word in item["question"].lower():
                score += 2
            if word in item["answer"].lower():
                score += 1
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [item for _, item in scored]
    return ECO_ASSISTANT_FAQ[:4]
