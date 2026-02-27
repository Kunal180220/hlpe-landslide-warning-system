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
    "Tamil Nadu": {
        "bbox": [76.2, 80.4, 8.0, 13.6],
        "color": "#e67e22",
        "landslide_prone": True,
        "districts": {
            "Nilgiris": {
                "bbox": [76.2, 77.0, 11.2, 11.8],
                "taluks": {
                    "Ooty (Udhagamandalam)": {"bbox": [76.6, 77.0, 11.3, 11.7], "center": [11.5, 76.8]},
                    "Coonoor": {"bbox": [76.7, 77.0, 11.3, 11.6], "center": [11.45, 76.85]},
                    "Gudalur": {"bbox": [76.2, 76.6, 11.4, 11.7], "center": [11.55, 76.4]},
                    "Pandalur": {"bbox": [76.3, 76.6, 11.5, 11.8], "center": [11.65, 76.45]},
                    "Kundah": {"bbox": [76.6, 76.95, 11.2, 11.5], "center": [11.35, 76.78]},
                }
            },
            "Coimbatore": {
                "bbox": [76.7, 77.5, 10.8, 11.3],
                "taluks": {
                    "Coimbatore North": {"bbox": [76.9, 77.2, 11.0, 11.3], "center": [11.15, 77.05]},
                    "Mettupalayam": {"bbox": [76.9, 77.2, 11.2, 11.5], "center": [11.3, 77.05]},
                    "Pollachi": {"bbox": [76.8, 77.2, 10.6, 11.0], "center": [10.8, 77.0]},
                    "Valparai": {"bbox": [76.9, 77.3, 10.3, 10.7], "center": [10.5, 77.1]},
                }
            },
            "Dindigul": {
                "bbox": [77.4, 78.0, 10.1, 10.6],
                "taluks": {
                    "Kodaikanal": {"bbox": [77.4, 77.7, 10.2, 10.5], "center": [10.35, 77.55]},
                    "Palani": {"bbox": [77.5, 77.9, 10.4, 10.7], "center": [10.55, 77.7]},
                    "Oddanchatram": {"bbox": [77.7, 78.0, 10.4, 10.7], "center": [10.55, 77.85]},
                }
            },
            "Tirunelveli": {
                "bbox": [77.0, 77.8, 8.5, 9.3],
                "taluks": {
                    "Ambasamudram": {"bbox": [77.4, 77.7, 8.7, 9.0], "center": [8.85, 77.55]},
                    "Kalakad": {"bbox": [77.5, 77.8, 8.4, 8.7], "center": [8.55, 77.65]},
                    "Tirunelveli": {"bbox": [77.6, 77.9, 8.6, 8.95], "center": [8.78, 77.75]},
                }
            },
        }
    },
    "Arunachal Pradesh": {
        "bbox": [91.5, 97.4, 26.6, 29.5],
        "color": "#16a085",
        "landslide_prone": True,
        "districts": {
            "West Kameng": {
                "bbox": [91.5, 92.8, 27.0, 28.0],
                "taluks": {
                    "Bomdila": {"bbox": [92.2, 92.6, 27.1, 27.5], "center": [27.3, 92.4]},
                    "Dirang": {"bbox": [92.3, 92.7, 27.4, 27.8], "center": [27.6, 92.5]},
                    "Kalaktang": {"bbox": [91.8, 92.3, 27.0, 27.4], "center": [27.2, 92.05]},
                }
            },
            "East Siang": {
                "bbox": [94.8, 95.8, 27.8, 28.6],
                "taluks": {
                    "Pasighat": {"bbox": [95.2, 95.6, 27.9, 28.3], "center": [28.1, 95.4]},
                    "Mebo": {"bbox": [94.9, 95.3, 28.1, 28.5], "center": [28.3, 95.1]},
                }
            },
            "Papum Pare": {
                "bbox": [93.0, 94.0, 27.0, 27.6],
                "taluks": {
                    "Itanagar": {"bbox": [93.5, 93.9, 27.0, 27.4], "center": [27.2, 93.7]},
                    "Doimukh": {"bbox": [93.6, 94.0, 27.1, 27.5], "center": [27.3, 93.8]},
                }
            },
        }
    },
    "Assam": {
        "bbox": [89.7, 96.0, 24.1, 28.2],
        "color": "#27ae60",
        "landslide_prone": True,
        "districts": {
            "Dima Hasao": {
                "bbox": [92.2, 93.4, 25.0, 26.0],
                "taluks": {
                    "Haflong": {"bbox": [92.9, 93.3, 25.0, 25.4], "center": [25.2, 93.1]},
                    "Maibang": {"bbox": [92.5, 92.9, 25.2, 25.6], "center": [25.4, 92.7]},
                    "Umrangso": {"bbox": [92.6, 93.0, 25.5, 25.9], "center": [25.7, 92.8]},
                }
            },
            "Karbi Anglong": {
                "bbox": [92.2, 93.8, 25.5, 26.7],
                "taluks": {
                    "Diphu": {"bbox": [93.2, 93.6, 25.7, 26.1], "center": [25.9, 93.4]},
                    "Bokajan": {"bbox": [93.4, 93.8, 26.0, 26.4], "center": [26.2, 93.6]},
                    "Hamren": {"bbox": [92.5, 92.9, 26.0, 26.4], "center": [26.2, 92.7]},
                }
            },
            "Cachar": {
                "bbox": [92.4, 93.2, 24.5, 25.2],
                "taluks": {
                    "Silchar": {"bbox": [92.7, 93.1, 24.7, 25.1], "center": [24.9, 92.9]},
                    "Sonai": {"bbox": [92.5, 92.9, 24.5, 24.9], "center": [24.7, 92.7]},
                }
            },
        }
    },
    "Manipur": {
        "bbox": [93.0, 94.8, 23.8, 25.7],
        "color": "#8e44ad",
        "landslide_prone": True,
        "districts": {
            "Senapati": {
                "bbox": [93.5, 94.3, 24.8, 25.5],
                "taluks": {
                    "Senapati": {"bbox": [93.8, 94.2, 25.0, 25.4], "center": [25.2, 94.0]},
                    "Paomata": {"bbox": [93.5, 93.9, 25.2, 25.6], "center": [25.4, 93.7]},
                    "Mao": {"bbox": [93.9, 94.3, 25.3, 25.7], "center": [25.5, 94.1]},
                }
            },
            "Churachandpur": {
                "bbox": [93.0, 93.9, 23.8, 24.6],
                "taluks": {
                    "Churachandpur": {"bbox": [93.5, 93.9, 24.2, 24.6], "center": [24.4, 93.7]},
                    "Tipaimukh": {"bbox": [93.0, 93.5, 23.8, 24.3], "center": [24.05, 93.25]},
                }
            },
            "Imphal West": {
                "bbox": [93.7, 94.1, 24.6, 25.0],
                "taluks": {
                    "Imphal": {"bbox": [93.8, 94.1, 24.7, 25.0], "center": [24.85, 93.95]},
                    "Patsoi": {"bbox": [93.7, 94.0, 24.8, 25.1], "center": [24.95, 93.85]},
                }
            },
        }
    },
    "Mizoram": {
        "bbox": [92.2, 93.5, 21.9, 24.5],
        "color": "#2980b9",
        "landslide_prone": True,
        "districts": {
            "Aizawl": {
                "bbox": [92.5, 93.1, 23.5, 24.1],
                "taluks": {
                    "Aizawl": {"bbox": [92.7, 93.0, 23.6, 23.9], "center": [23.75, 92.85]},
                    "Darlawn": {"bbox": [92.6, 92.95, 23.8, 24.1], "center": [23.95, 92.78]},
                }
            },
            "Lunglei": {
                "bbox": [92.5, 93.2, 22.5, 23.3],
                "taluks": {
                    "Lunglei": {"bbox": [92.7, 93.1, 22.7, 23.1], "center": [22.9, 92.9]},
                    "Hnahthial": {"bbox": [92.8, 93.2, 23.0, 23.4], "center": [23.2, 93.0]},
                }
            },
        }
    },
    "Nagaland": {
        "bbox": [93.2, 95.3, 25.2, 27.1],
        "color": "#c0392b",
        "landslide_prone": True,
        "districts": {
            "Kohima": {
                "bbox": [93.8, 94.4, 25.5, 26.1],
                "taluks": {
                    "Kohima": {"bbox": [94.0, 94.4, 25.6, 26.0], "center": [25.8, 94.2]},
                    "Zubza": {"bbox": [93.9, 94.3, 25.7, 26.1], "center": [25.9, 94.1]},
                }
            },
            "Phek": {
                "bbox": [94.2, 95.0, 25.5, 26.3],
                "taluks": {
                    "Phek": {"bbox": [94.4, 94.8, 25.7, 26.1], "center": [25.9, 94.6]},
                    "Pfutsero": {"bbox": [94.6, 95.0, 25.8, 26.2], "center": [26.0, 94.8]},
                }
            },
            "Tuensang": {
                "bbox": [94.5, 95.3, 26.0, 26.8],
                "taluks": {
                    "Tuensang": {"bbox": [94.7, 95.1, 26.1, 26.5], "center": [26.3, 94.9]},
                    "Noklak": {"bbox": [94.9, 95.3, 26.3, 26.7], "center": [26.5, 95.1]},
                }
            },
        }
    },
    "Sikkim": {
        "bbox": [88.0, 88.9, 27.0, 28.1],
        "color": "#1abc9c",
        "landslide_prone": True,
        "districts": {
            "East Sikkim": {
                "bbox": [88.4, 88.9, 27.1, 27.6],
                "taluks": {
                    "Gangtok": {"bbox": [88.5, 88.8, 27.2, 27.5], "center": [27.35, 88.65]},
                    "Pakyong": {"bbox": [88.6, 88.9, 27.0, 27.3], "center": [27.15, 88.75]},
                    "Rongli": {"bbox": [88.7, 88.95, 27.2, 27.5], "center": [27.35, 88.83]},
                }
            },
            "North Sikkim": {
                "bbox": [88.0, 88.8, 27.6, 28.1],
                "taluks": {
                    "Mangan": {"bbox": [88.3, 88.7, 27.5, 27.9], "center": [27.7, 88.5]},
                    "Chungthang": {"bbox": [88.2, 88.6, 27.7, 28.1], "center": [27.9, 88.4]},
                }
            },
            "West Sikkim": {
                "bbox": [88.0, 88.5, 27.1, 27.7],
                "taluks": {
                    "Gyalshing": {"bbox": [88.1, 88.5, 27.2, 27.6], "center": [27.4, 88.3]},
                    "Soreng": {"bbox": [88.0, 88.4, 27.0, 27.4], "center": [27.2, 88.2]},
                }
            },
            "South Sikkim": {
                "bbox": [88.2, 88.7, 27.0, 27.4],
                "taluks": {
                    "Namchi": {"bbox": [88.3, 88.6, 27.1, 27.4], "center": [27.25, 88.45]},
                    "Ravangla": {"bbox": [88.3, 88.6, 27.2, 27.5], "center": [27.35, 88.45]},
                }
            },
        }
    },
    "Tripura": {
        "bbox": [91.1, 92.3, 22.9, 24.5],
        "color": "#d35400",
        "landslide_prone": True,
        "districts": {
            "Dhalai": {
                "bbox": [91.8, 92.3, 23.6, 24.2],
                "taluks": {
                    "Ambassa": {"bbox": [91.8, 92.1, 23.8, 24.1], "center": [23.95, 91.95]},
                    "Kamalpur": {"bbox": [91.9, 92.2, 24.0, 24.3], "center": [24.15, 92.05]},
                }
            },
            "North Tripura": {
                "bbox": [91.7, 92.3, 23.9, 24.5],
                "taluks": {
                    "Dharmanagar": {"bbox": [92.0, 92.3, 24.2, 24.5], "center": [24.35, 92.15]},
                    "Kanchanpur": {"bbox": [91.9, 92.2, 24.1, 24.4], "center": [24.25, 92.05]},
                }
            },
        }
    },
    "West Bengal": {
        "bbox": [85.8, 89.9, 21.4, 27.2],
        "color": "#f39c12",
        "landslide_prone": True,
        "districts": {
            "Darjeeling": {
                "bbox": [87.9, 88.9, 26.7, 27.2],
                "taluks": {
                    "Darjeeling Sadar": {"bbox": [88.2, 88.5, 27.0, 27.2], "center": [27.1, 88.35]},
                    "Kurseong": {"bbox": [88.2, 88.5, 26.8, 27.1], "center": [26.95, 88.35]},
                    "Kalimpong": {"bbox": [88.4, 88.8, 27.0, 27.2], "center": [27.1, 88.6]},
                    "Mirik": {"bbox": [88.0, 88.3, 26.8, 27.1], "center": [26.95, 88.15]},
                    "Siliguri": {"bbox": [88.2, 88.5, 26.6, 26.9], "center": [26.75, 88.35]},
                }
            },
            "Kalimpong": {
                "bbox": [88.4, 88.9, 27.0, 27.4],
                "taluks": {
                    "Kalimpong I": {"bbox": [88.4, 88.7, 27.0, 27.3], "center": [27.15, 88.55]},
                    "Kalimpong II": {"bbox": [88.6, 88.9, 27.1, 27.4], "center": [27.25, 88.75]},
                    "Gorubathan": {"bbox": [88.5, 88.8, 26.8, 27.1], "center": [26.95, 88.65]},
                }
            },
            "Jalpaiguri": {
                "bbox": [88.5, 89.2, 26.3, 27.0],
                "taluks": {
                    "Jalpaiguri": {"bbox": [88.6, 88.9, 26.4, 26.7], "center": [26.55, 88.75]},
                    "Mal": {"bbox": [88.9, 89.2, 26.6, 26.9], "center": [26.75, 89.05]},
                    "Alipurduar": {"bbox": [89.3, 89.6, 26.4, 26.7], "center": [26.55, 89.45]},
                }
            },
        }
    },
    "Jammu & Kashmir": {
        "bbox": [73.7, 80.4, 32.3, 37.1],
        "color": "#2c3e50",
        "landslide_prone": True,
        "districts": {
            "Ramban": {
                "bbox": [75.0, 75.7, 33.0, 33.5],
                "taluks": {
                    "Ramban": {"bbox": [75.2, 75.6, 33.1, 33.4], "center": [33.25, 75.4]},
                    "Banihal": {"bbox": [75.1, 75.5, 33.3, 33.6], "center": [33.45, 75.3]},
                    "Gool": {"bbox": [75.3, 75.7, 33.2, 33.5], "center": [33.35, 75.5]},
                }
            },
            "Doda": {
                "bbox": [75.5, 76.4, 33.0, 33.7],
                "taluks": {
                    "Doda": {"bbox": [75.8, 76.2, 33.0, 33.4], "center": [33.2, 76.0]},
                    "Bhaderwah": {"bbox": [75.6, 76.0, 33.1, 33.5], "center": [33.3, 75.8]},
                    "Thathri": {"bbox": [75.9, 76.3, 33.3, 33.7], "center": [33.5, 76.1]},
                }
            },
            "Reasi": {
                "bbox": [74.6, 75.2, 32.8, 33.3],
                "taluks": {
                    "Reasi": {"bbox": [74.7, 75.1, 33.0, 33.3], "center": [33.15, 74.9]},
                    "Mahore": {"bbox": [74.6, 75.0, 33.1, 33.4], "center": [33.25, 74.8]},
                }
            },
            "Poonch": {
                "bbox": [73.9, 74.5, 33.6, 34.1],
                "taluks": {
                    "Poonch": {"bbox": [74.0, 74.4, 33.7, 34.0], "center": [33.85, 74.2]},
                    "Surankote": {"bbox": [74.1, 74.5, 33.8, 34.1], "center": [33.95, 74.3]},
                }
            },
        }
    },
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
