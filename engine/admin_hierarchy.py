"""
Administrative Hierarchy Module
Provides State → District → Taluk level data and bounding boxes
for India with focus on landslide-prone regions.
"""

import json
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# INDIA ADMINISTRATIVE HIERARCHY
# Focused on landslide-prone states. Extend as needed.
# bbox format: [min_lon, max_lon, min_lat, max_lat]
# ─────────────────────────────────────────────────────────────────────────────

ADMIN_HIERARCHY: Dict = {
    "Kerala": {
        "bbox": [74.8, 77.4, 8.3, 12.8],
        "color": "#2ecc71",
        "landslide_prone": True,
        "districts": {
            "Idukki": {
                "bbox": [76.7, 77.4, 9.8, 10.4],
                "taluks": {
                    "Devikulam": {"bbox": [77.0, 77.4, 10.0, 10.3], "center": [10.15, 77.2]},
                    "Udumpanpara": {"bbox": [76.8, 77.1, 9.9, 10.2], "center": [10.05, 76.95]},
                    "Peerumedu": {"bbox": [76.9, 77.3, 9.7, 10.0], "center": [9.85, 77.1]},
                    "Idukki": {"bbox": [76.7, 77.0, 9.8, 10.1], "center": [9.95, 76.85]},
                    "Thodupuzha": {"bbox": [76.7, 76.95, 9.9, 10.2], "center": [10.05, 76.82]},
                }
            },
            "Wayanad": {
                "bbox": [75.7, 76.4, 11.4, 12.0],
                "taluks": {
                    "Mananthavady": {"bbox": [75.9, 76.3, 11.7, 12.0], "center": [11.85, 76.1]},
                    "Sulthan Bathery": {"bbox": [76.1, 76.4, 11.6, 11.9], "center": [11.75, 76.25]},
                    "Vythiri": {"bbox": [75.7, 76.1, 11.4, 11.7], "center": [11.55, 75.9]},
                    "Kalpetta": {"bbox": [75.9, 76.2, 11.5, 11.8], "center": [11.65, 76.05]},
                }
            },
            "Malappuram": {
                "bbox": [75.9, 76.5, 10.8, 11.3],
                "taluks": {
                    "Nilambur": {"bbox": [76.1, 76.5, 11.0, 11.3], "center": [11.15, 76.3]},
                    "Tirur": {"bbox": [75.9, 76.2, 10.8, 11.1], "center": [10.95, 76.05]},
                    "Perinthalmanna": {"bbox": [76.2, 76.5, 10.9, 11.2], "center": [11.05, 76.35]},
                }
            },
            "Kozhikode": {
                "bbox": [75.7, 76.2, 11.1, 11.6],
                "taluks": {
                    "Kozhikode": {"bbox": [75.7, 76.0, 11.1, 11.4], "center": [11.25, 75.85]},
                    "Vadakara": {"bbox": [75.8, 76.1, 11.4, 11.7], "center": [11.55, 75.95]},
                    "Thamarassery": {"bbox": [76.0, 76.2, 11.3, 11.6], "center": [11.45, 76.1]},
                }
            },
        }
    },
    "Karnataka": {
        "bbox": [74.0, 78.6, 11.6, 18.5],
        "color": "#e74c3c",
        "landslide_prone": True,
        "districts": {
            "Kodagu": {
                "bbox": [75.5, 76.4, 11.8, 12.8],
                "taluks": {
                    "Madikeri": {"bbox": [75.6, 75.95, 12.3, 12.7], "center": [12.5, 75.75]},
                    "Virajpet": {"bbox": [75.8, 76.3, 11.9, 12.3], "center": [12.1, 76.05]},
                    "Somwarpet": {"bbox": [75.9, 76.4, 12.3, 12.8], "center": [12.55, 76.15]},
                }
            },
            "Chikkamagaluru": {
                "bbox": [75.6, 76.4, 13.1, 13.9],
                "taluks": {
                    "Chikkamagaluru": {"bbox": [75.6, 76.0, 13.5, 13.9], "center": [13.7, 75.8]},
                    "Mudigere": {"bbox": [75.6, 75.9, 13.1, 13.5], "center": [13.3, 75.75]},
                    "Koppa": {"bbox": [75.3, 75.7, 13.4, 13.8], "center": [13.6, 75.5]},
                    "Sringeri": {"bbox": [75.1, 75.4, 13.4, 13.7], "center": [13.55, 75.25]},
                }
            },
            "Uttara Kannada": {
                "bbox": [74.1, 75.3, 14.0, 15.3],
                "taluks": {
                    "Sirsi": {"bbox": [74.8, 75.2, 14.6, 15.0], "center": [14.8, 75.0]},
                    "Yellapur": {"bbox": [74.7, 75.1, 14.9, 15.3], "center": [15.1, 74.9]},
                    "Siddapur": {"bbox": [74.9, 75.3, 14.3, 14.7], "center": [14.5, 75.1]},
                }
            },
        }
    },
    "Himachal Pradesh": {
        "bbox": [75.5, 79.0, 30.4, 33.2],
        "color": "#3498db",
        "landslide_prone": True,
        "districts": {
            "Mandi": {
                "bbox": [76.6, 77.6, 31.5, 32.1],
                "taluks": {
                    "Mandi Sadar": {"bbox": [76.8, 77.2, 31.6, 32.0], "center": [31.8, 77.0]},
                    "Sundernagar": {"bbox": [76.7, 77.1, 31.5, 31.9], "center": [31.7, 76.9]},
                    "Jogindernagar": {"bbox": [76.6, 77.0, 31.9, 32.1], "center": [32.0, 76.8]},
                }
            },
            "Kullu": {
                "bbox": [77.0, 78.0, 31.8, 32.6],
                "taluks": {
                    "Kullu": {"bbox": [77.0, 77.5, 31.9, 32.3], "center": [32.1, 77.25]},
                    "Manali": {"bbox": [77.1, 77.6, 32.2, 32.6], "center": [32.4, 77.35]},
                    "Banjar": {"bbox": [77.3, 77.8, 31.8, 32.1], "center": [31.95, 77.55]},
                }
            },
            "Shimla": {
                "bbox": [77.0, 78.4, 30.8, 31.7],
                "taluks": {
                    "Shimla Urban": {"bbox": [77.0, 77.3, 31.0, 31.3], "center": [31.15, 77.15]},
                    "Theog": {"bbox": [77.3, 77.7, 31.1, 31.5], "center": [31.3, 77.5]},
                    "Rampur": {"bbox": [77.6, 78.2, 31.3, 31.7], "center": [31.5, 77.9]},
                }
            },
        }
    },
    "Uttarakhand": {
        "bbox": [77.5, 81.1, 28.7, 31.5],
        "color": "#9b59b6",
        "landslide_prone": True,
        "districts": {
            "Chamoli": {
                "bbox": [79.0, 80.6, 30.0, 31.2],
                "taluks": {
                    "Gopeshwar": {"bbox": [79.3, 79.7, 30.4, 30.8], "center": [30.6, 79.5]},
                    "Joshimath": {"bbox": [79.5, 79.9, 30.5, 30.9], "center": [30.7, 79.7]},
                    "Tharali": {"bbox": [79.0, 79.5, 30.0, 30.4], "center": [30.2, 79.25]},
                }
            },
            "Rudraprayag": {
                "bbox": [78.6, 79.4, 30.3, 31.0],
                "taluks": {
                    "Rudraprayag": {"bbox": [78.8, 79.2, 30.5, 30.9], "center": [30.7, 79.0]},
                    "Ukhimath": {"bbox": [78.6, 79.0, 30.6, 31.0], "center": [30.8, 78.8]},
                }
            },
            "Tehri Garhwal": {
                "bbox": [78.0, 79.2, 30.2, 30.8],
                "taluks": {
                    "Tehri": {"bbox": [78.4, 78.8, 30.4, 30.8], "center": [30.6, 78.6]},
                    "Narendra Nagar": {"bbox": [78.2, 78.6, 30.2, 30.6], "center": [30.4, 78.4]},
                    "Pratapnagar": {"bbox": [78.7, 79.2, 30.3, 30.7], "center": [30.5, 78.95]},
                }
            },
        }
    },
    "Maharashtra": {
        "bbox": [72.6, 80.9, 15.6, 22.1],
        "color": "#f39c12",
        "landslide_prone": True,
        "districts": {
            "Raigad": {
                "bbox": [72.9, 73.7, 18.0, 18.8],
                "taluks": {
                    "Mahad": {"bbox": [73.3, 73.7, 18.0, 18.4], "center": [18.2, 73.5]},
                    "Poladpur": {"bbox": [73.3, 73.7, 17.9, 18.3], "center": [18.1, 73.5]},
                    "Sudhagad": {"bbox": [73.0, 73.4, 18.4, 18.8], "center": [18.6, 73.2]},
                }
            },
            "Satara": {
                "bbox": [73.5, 74.4, 17.4, 18.2],
                "taluks": {
                    "Satara": {"bbox": [73.9, 74.4, 17.6, 18.0], "center": [17.8, 74.15]},
                    "Mahabaleshwar": {"bbox": [73.6, 74.0, 17.8, 18.2], "center": [18.0, 73.8]},
                    "Javali": {"bbox": [73.7, 74.2, 17.5, 17.9], "center": [17.7, 73.95]},
                }
            },
        }
    },
    "Meghalaya": {
        "bbox": [89.8, 92.8, 24.9, 26.1],
        "color": "#1abc9c",
        "landslide_prone": True,
        "districts": {
            "East Khasi Hills": {
                "bbox": [91.5, 92.2, 25.3, 25.8],
                "taluks": {
                    "Shillong": {"bbox": [91.8, 92.1, 25.5, 25.8], "center": [25.65, 91.95]},
                    "Cherrapunjee": {"bbox": [91.6, 91.9, 25.3, 25.6], "center": [25.45, 91.75]},
                    "Mawkyrwat": {"bbox": [91.5, 91.8, 25.4, 25.7], "center": [25.55, 91.65]},
                }
            },
            "West Khasi Hills": {
                "bbox": [90.4, 91.5, 25.1, 25.8],
                "taluks": {
                    "Nongstoin": {"bbox": [91.0, 91.4, 25.4, 25.8], "center": [25.6, 91.2]},
                    "Mawkyrwat": {"bbox": [90.6, 91.0, 25.1, 25.5], "center": [25.3, 90.8]},
                }
            },
        }
    },
}


def get_states() -> List[str]:
    return sorted(ADMIN_HIERARCHY.keys())


def get_districts(state: str) -> List[str]:
    if state not in ADMIN_HIERARCHY:
        return []
    return sorted(ADMIN_HIERARCHY[state]["districts"].keys())


def get_taluks(state: str, district: str) -> List[str]:
    try:
        return sorted(ADMIN_HIERARCHY[state]["districts"][district]["taluks"].keys())
    except KeyError:
        return []


def get_bbox(state: str, district: Optional[str] = None, taluk: Optional[str] = None) -> List[float]:
    """Returns [min_lon, max_lon, min_lat, max_lat]"""
    try:
        if taluk and district:
            t = ADMIN_HIERARCHY[state]["districts"][district]["taluks"][taluk]
            return t["bbox"]
        elif district:
            return ADMIN_HIERARCHY[state]["districts"][district]["bbox"]
        else:
            return ADMIN_HIERARCHY[state]["bbox"]
    except KeyError:
        return [75.0, 77.5, 10.0, 12.0]  # default Kerala


def get_center(state: str, district: Optional[str] = None, taluk: Optional[str] = None) -> Tuple[float, float]:
    """Returns (lat, lon) center"""
    bbox = get_bbox(state, district, taluk)
    if taluk and district:
        try:
            t = ADMIN_HIERARCHY[state]["districts"][district]["taluks"][taluk]
            if "center" in t:
                return tuple(t["center"])
        except KeyError:
            pass
    lat = (bbox[2] + bbox[3]) / 2
    lon = (bbox[0] + bbox[1]) / 2
    return (lat, lon)


def get_admin_info(state: str, district: Optional[str] = None, taluk: Optional[str] = None) -> Dict:
    """Returns full info for the selected admin level"""
    info = {
        "level": "state",
        "name": state,
        "bbox": get_bbox(state),
        "center": get_center(state),
        "zoom": 7,
    }
    if district:
        info.update({
            "level": "district",
            "name": district,
            "parent": state,
            "bbox": get_bbox(state, district),
            "center": get_center(state, district),
            "zoom": 10,
        })
    if taluk and district:
        info.update({
            "level": "taluk",
            "name": taluk,
            "parent": district,
            "bbox": get_bbox(state, district, taluk),
            "center": get_center(state, district, taluk),
            "zoom": 12,
        })
    return info


def get_all_taluks_in_district(state: str, district: str) -> Dict:
    """Returns all taluk data for a district"""
    try:
        return ADMIN_HIERARCHY[state]["districts"][district]["taluks"]
    except KeyError:
        return {}


def get_landslide_prone_states() -> List[str]:
    return [s for s, d in ADMIN_HIERARCHY.items() if d.get("landslide_prone")]
