"""Global Municipal Waste Segregation Protocols & Bin Color Standards.

Provides regional municipal bin allocations, color hex codes, and local disposal
guidelines for the 12 material categories across 5 major global jurisdictions:
- United States (Single-Stream Standard)
- European Union (Circular Dual-System Standard / Green Dot)
- United Kingdom (Local Council Recycling Protocol)
- Japan (Strict Multi-Stream Resource Segregation)
- India (Swachh Bharat Urban Solid Waste Rules)
"""
from typing import Any, Dict

MUNICIPAL_JURISDICTIONS: Dict[str, Dict[str, Any]] = {
    "🇺🇸 United States (Single-Stream Standard)": {
        "region_code": "US",
        "description": "Standard American curbside single-stream recycling with blue bins and separate organics/trash.",
        "bins": {
            "battery": {
                "bin_name": "Hazardous E-Waste Drop-off (Call2Recycle / Retail)",
                "bin_color": "#dc2626",
                "rules": "Never place in curbside carts. Drop off at Home Depot, Best Buy, or county household hazardous waste (HHW) sites.",
            },
            "biological": {
                "bin_name": "Green Cart (Organics & Yard Trimmings)",
                "bin_color": "#15803d",
                "rules": "Food scraps, soiled paper napkins, and landscape trimmings. Keep free of conventional plastic bags.",
            },
            "brown-glass": {
                "bin_name": "Blue Cart (Mixed Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Empty and rinse. Some municipalities require separate glass crates to prevent shattering in single-stream.",
            },
            "cardboard": {
                "bin_name": "Blue Cart (Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Flatten completely. Grease-free boxes only; cut out greasy pizza box sections.",
            },
            "clothes": {
                "bin_name": "Textile Donation Box / Goodwill",
                "bin_color": "#d97706",
                "rules": "Clean wearable garments to donation bins. Heavily torn/stained textiles to textile recycler.",
            },
            "green-glass": {
                "bin_name": "Blue Cart (Mixed Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Rinse clean. Do not mix with ceramic mugs or lightbulbs.",
            },
            "metal": {
                "bin_name": "Blue Cart (Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Aluminium beverage cans and steel food cans. Rinse out food residue.",
            },
            "paper": {
                "bin_name": "Blue Cart (Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Office paper, magazines, envelopes. Keep dry; shredded paper should be contained in paper bag if allowed.",
            },
            "plastic": {
                "bin_name": "Blue Cart (Rigid Plastics #1 & #2)",
                "bin_color": "#2563eb",
                "rules": "Bottles and jugs. Film plastics (grocery bags) return to supermarket store drop-offs.",
            },
            "shoes": {
                "bin_name": "Donation Bin / Shoe Recycling (Nike Grind)",
                "bin_color": "#d97706",
                "rules": "Tie in pairs. Unwearable footwear to specialized athletic shoe collection points or black trash.",
            },
            "trash": {
                "bin_name": "Black / Gray Cart (Landfill Residual)",
                "bin_color": "#4b5563",
                "rules": "General municipal solid waste destined for landfill or waste-to-energy.",
            },
            "white-glass": {
                "bin_name": "Blue Cart (Mixed Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Clear jars and bottles. Rinse cleanly.",
            },
        },
    },
    "🇪🇺 European Union (Circular Dual-System / Pfand)": {
        "region_code": "EU",
        "description": "European standard: Yellow packaging, Blue paper, Green/Brown bio, and Glass stream with deposit refund.",
        "bins": {
            "battery": {
                "bin_name": "Batterie-Sammelbox (Supermarket Return Box)",
                "bin_color": "#b91c1c",
                "rules": "Legal take-back in all retail supermarkets and electronics shops (BattG). Tape lithium terminals.",
            },
            "biological": {
                "bin_name": "Biotonne (Brown Bin - Organic Waste)",
                "bin_color": "#78350f",
                "rules": "Kitchen peels, coffee grounds, garden clippings. Only paper bags or certified compostable liners.",
            },
            "brown-glass": {
                "bin_name": "Braunglas-Container (Brown Glass Bottle Bank)",
                "bin_color": "#92400e",
                "rules": "Separate curbside igloo for brown glass. Observe municipal noise quiet hours (e.g. 8am-8pm).",
            },
            "cardboard": {
                "bin_name": "Blaue Tonne (Blue Bin - Paper & Cardboard)",
                "bin_color": "#1d4ed8",
                "rules": "Flatten packaging cartons. Waxed or heavily plastic-laminated cardboard is excluded.",
            },
            "clothes": {
                "bin_name": "Altkleidercontainer (Charity Textile Bank)",
                "bin_color": "#ea580c",
                "rules": "Enclosed in closed plastic bags for collection by Red Cross or local charities.",
            },
            "green-glass": {
                "bin_name": "Grünglas-Container (Green Glass Bottle Bank)",
                "bin_color": "#15803d",
                "rules": "Green and other colored glass (blue, red). Empty and lidless.",
            },
            "metal": {
                "bin_name": "Gelber Sack / Gelbe Tonne (Yellow Packaging Bin)",
                "bin_color": "#eab308",
                "rules": "Aluminium tins, food cans, metal bottle caps. Check for 0.25€ Pfand return on drink cans.",
            },
            "paper": {
                "bin_name": "Blaue Tonne (Blue Bin - Paper)",
                "bin_color": "#1d4ed8",
                "rules": "Newspapers, packaging paper, catalogs. No thermal receipts or waxed gift wrap.",
            },
            "plastic": {
                "bin_name": "Gelbe Tonne / Wertstofftonne (Yellow Bin)",
                "bin_color": "#eab308",
                "rules": "Lightweight packaging (Grüner Punkt). Single-use PET beverage bottles return to Pfand machine (0.25€ refund).",
            },
            "shoes": {
                "bin_name": "Altkleidercontainer (Paired Shoes)",
                "bin_color": "#ea580c",
                "rules": "Tie pairs together. Non-usable shoes go to Restmüll (Residual).",
            },
            "trash": {
                "bin_name": "Restmüll (Black/Gray Bin - Residual Waste)",
                "bin_color": "#374151",
                "rules": "Non-recyclable items. Subject to volume-based municipal waste fees.",
            },
            "white-glass": {
                "bin_name": "Weißglas-Container (Clear Glass Bottle Bank)",
                "bin_color": "#0284c7",
                "rules": "Exclusively transparent, uncolored glass jars and bottles.",
            },
        },
    },
    "🇬🇧 United Kingdom (Local Council Recycling Protocol)": {
        "region_code": "UK",
        "description": "UK Council system with mixed dry recyclables, dedicated food waste caddies, and HWRC centres.",
        "bins": {
            "battery": {
                "bin_name": "Supermarket Battery Tube / HWRC Centre",
                "bin_color": "#dc2626",
                "rules": "Deposit in battery collection tubes at Tesco, Sainsbury's, or local HWRC council tip.",
            },
            "biological": {
                "bin_name": "Brown Food Waste Caddy (Organics)",
                "bin_color": "#854d0e",
                "rules": "Plate scrapings, fruit peels, teabags. Use compostable cornstarch liners.",
            },
            "brown-glass": {
                "bin_name": "Green / Blue Box (Glass & Recyclables)",
                "bin_color": "#2563eb",
                "rules": "Rinse clean. Place in council glass box or mixed recycling wheelie bin.",
            },
            "cardboard": {
                "bin_name": "Blue/Green Wheelie Bin (Dry Mixed Recycling)",
                "bin_color": "#16a34a",
                "rules": "Flatten packaging. Remove bubble wrap and polystyrene packing peanuts.",
            },
            "clothes": {
                "bin_name": "Charity Shop / Council Textile Bank",
                "bin_color": "#ea580c",
                "rules": "Donate to Oxfam, British Heart Foundation, or council recycling centre.",
            },
            "green-glass": {
                "bin_name": "Green/Blue Box (Glass Collection)",
                "bin_color": "#2563eb",
                "rules": "Rinse bottles and jars. Metal caps can often stay on.",
            },
            "metal": {
                "bin_name": "Green/Blue Wheelie Bin (Tins & Cans)",
                "bin_color": "#16a34a",
                "rules": "Drink cans, food tins, clean aluminium foil. Scrunch foil into a ball.",
            },
            "paper": {
                "bin_name": "Blue/Green Wheelie Bin (Paper)",
                "bin_color": "#16a34a",
                "rules": "Junk mail, newspapers, cardboard boxes. No grease or takeaway grease stains.",
            },
            "plastic": {
                "bin_name": "Green/Blue Wheelie Bin (Plastic Bottles & Pots)",
                "bin_color": "#16a34a",
                "rules": "Bottles, pots, tubs, and trays. Soft flexible plastic films return to supermarkets.",
            },
            "shoes": {
                "bin_name": "Shoe Bank / Charity Shop",
                "bin_color": "#ea580c",
                "rules": "Wearable pairs to charity. Damaged pairs to Black General Waste Bin.",
            },
            "trash": {
                "bin_name": "Black / Grey Wheelie Bin (General Refuse)",
                "bin_color": "#4b5563",
                "rules": "Non-recyclable household waste collected bi-weekly.",
            },
            "white-glass": {
                "bin_name": "Green/Blue Box (Glass Collection)",
                "bin_color": "#2563eb",
                "rules": "Clear food jars and bottles. Rinse before depositing.",
            },
        },
    },
    "🇯🇵 Japan (Strict Multi-Stream Segregation)": {
        "region_code": "JP",
        "description": "Japanese municipal system: Moeru (burnable), Moenai (non-burnable), and meticulous Shigen (resource) sorting.",
        "bins": {
            "battery": {
                "bin_name": "Yūgai Gomi (Hazardous / Municipal Station)",
                "bin_color": "#dc2626",
                "rules": "Special collection days or designated drop-off boxes at ward offices and convenience stores.",
            },
            "biological": {
                "bin_name": "Moeru Gomi (Burnable Waste - Yellow Bag)",
                "bin_color": "#eab308",
                "rules": "Drain kitchen waste thoroughly of liquid before disposal in designated municipal bags.",
            },
            "brown-glass": {
                "bin_name": "Bin (Glass Bottles - Brown Crate)",
                "bin_color": "#92400e",
                "rules": "Rinse, remove cap, deposit in brown collection crate at neighbourhood station.",
            },
            "cardboard": {
                "bin_name": "Danbōru (Cardboard - Tied with Twine)",
                "bin_color": "#2563eb",
                "rules": "Flatten and bundle tightly with paper string/twine on designated paper collection days.",
            },
            "clothes": {
                "bin_name": "Kofuku (Used Clothing Collection)",
                "bin_color": "#ea580c",
                "rules": "Clean clothes in transparent bags on dry resource collection days.",
            },
            "green-glass": {
                "bin_name": "Bin (Glass Bottles - Other Colors Crate)",
                "bin_color": "#15803d",
                "rules": "Wash clean and place in designated colored glass crate.",
            },
            "metal": {
                "bin_name": "Kan (Beverage & Food Cans)",
                "bin_color": "#0284c7",
                "rules": "Rinse and crush aluminium cans; separate steel from aluminium if specified by ward.",
            },
            "paper": {
                "bin_name": "Kosshi (Paper Resources - Bundled)",
                "bin_color": "#2563eb",
                "rules": "Newspapers, flyers, milk cartons (washed, cut open, dried, and tied).",
            },
            "plastic": {
                "bin_name": "Pura (Plastic Packaging with Mark)",
                "bin_color": "#06b6d4",
                "rules": "PET bottles: remove cap & label (go to Pura), rinse bottle body (goes to PET basket).",
            },
            "shoes": {
                "bin_name": "Moenai Gomi (Non-burnable) or Moeru Gomi",
                "bin_color": "#4b5563",
                "rules": "Check ward rules (some accept leather in burnable; rubber/synthetic in non-burnable).",
            },
            "trash": {
                "bin_name": "Moeru / Moenai Gomi (Municipal Bags)",
                "bin_color": "#4b5563",
                "rules": "Disposed in ward-sanctioned transparent bags on designated weekday mornings.",
            },
            "white-glass": {
                "bin_name": "Mumei Bin (Clear Glass Bottles Crate)",
                "bin_color": "#38bdf8",
                "rules": "Transparent glass containers. Separate from heat-resistant glassware.",
            },
        },
    },
    "🇮🇳 India (Swachh Bharat Urban Solid Waste Rules)": {
        "region_code": "IN",
        "description": "Indian standard (SBM 2.0): Green (Wet/Compost), Blue (Dry Recyclable), and Red (Domestic Hazardous).",
        "bins": {
            "battery": {
                "bin_name": "Red Bin (Domestic Hazardous Waste)",
                "bin_color": "#dc2626",
                "rules": "Hand over to authorized e-waste kabadiwala or municipal hazardous waste vehicle.",
            },
            "biological": {
                "bin_name": "Green Bin (Wet Waste / Gila Kachra)",
                "bin_color": "#15803d",
                "rules": "Organic food waste, vegetable peels, flower garlands. Compost directly or hand to municipal wet collector.",
            },
            "brown-glass": {
                "bin_name": "Blue Bin (Dry Waste / Sookha Kachra)",
                "bin_color": "#2563eb",
                "rules": "Rinse bottles. Wrap broken glass securely in newspaper to protect sanitation workers.",
            },
            "cardboard": {
                "bin_name": "Blue Bin (Dry Waste / Raddi)",
                "bin_color": "#2563eb",
                "rules": "Keep dry and flatten boxes. Can be sold to local dry waste collection centres or raddiwala.",
            },
            "clothes": {
                "bin_name": "Reuse / Donate (Goonj / Local NGOs)",
                "bin_color": "#d97706",
                "rules": "Donate clean clothes to donation drives or repurpose as cleaning dusters.",
            },
            "green-glass": {
                "bin_name": "Blue Bin (Dry Waste / Glass Stream)",
                "bin_color": "#2563eb",
                "rules": "Empty glass beverage bottles. Keep separate from ceramic cups.",
            },
            "metal": {
                "bin_name": "Blue Bin (Dry Recyclables / Bhangar)",
                "bin_color": "#2563eb",
                "rules": "Aluminium and steel tins. High scrap value at scrap dealers.",
            },
            "paper": {
                "bin_name": "Blue Bin (Dry Waste / Raddi)",
                "bin_color": "#2563eb",
                "rules": "Newspapers, notebooks, cardboard. Keep away from water and oil.",
            },
            "plastic": {
                "bin_name": "Blue Bin (Dry Waste - Plastics)",
                "bin_color": "#2563eb",
                "rules": "Clean containers, milk packets (cut corner partially to prevent microplastic litter).",
            },
            "shoes": {
                "bin_name": "Cobbler Repair / Dry Reject Waste",
                "bin_color": "#4b5563",
                "rules": "Repair at local cobbler (mochi) or donate wearable footwear.",
            },
            "trash": {
                "bin_name": "Black Bin (Sanitary / Domestic Reject Waste)",
                "bin_color": "#374151",
                "rules": "Non-recyclable sweeps and rejects sent to municipal processing plant.",
            },
            "white-glass": {
                "bin_name": "Blue Bin (Dry Waste - Clear Glass)",
                "bin_color": "#2563eb",
                "rules": "Clean jars (pickle, jam). High recycling value.",
            },
        },
    },
}


def get_jurisdiction_info(jurisdiction_name: str, class_key: str) -> Dict[str, str]:
    """Retrieve the bin name, bin color, and rules for a specific material and region."""
    jur = MUNICIPAL_JURISDICTIONS.get(jurisdiction_name, MUNICIPAL_JURISDICTIONS["🇺🇸 United States (Single-Stream Standard)"])
    return jur["bins"].get(class_key, {
        "bin_name": "Designated Recyclable Container",
        "bin_color": "#10b981",
        "rules": "Segregate according to local rules.",
    })
