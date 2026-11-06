"""The photo series of the Muzeul Olteniei Craiova set: folder (Romanian) -> object name in English,
setting, detector prompt and shape. Writes series.json (one row per photo folder)."""
import json
import sys
from pathlib import Path
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:/MASTER/Poze pentru disertatie")
# object key: (Romanian folder name prefixes, English label, detector prompt, shape: upright | flat)
OBJECTS = {
    "vessel_bird_protomes_001": (["001"], "black vessel with bird-shaped protomes", "vase", "upright"),
    "vessel_incised_002": (["002"], "incised vessel with white inlay", "vase", "upright"),
    "idol_bird_head_003": (["003"], "bell-shaped idol with a bird head", "figurine", "upright"),
    "vessel_carna_1427": (["1427 Carna Vas ceramic"], "ceramic vessel from Carna", "vase", "upright"),
    "fibula_bronze_55790": (["55790 Fibula bronz"], "bronze fibula (brooch)", "brooch", "flat"),
    "bracelet_spiral_55812": (["55812 Bratara plurispiralica"], "spiral bracelet", "bracelet", "flat"),
    "ring_bronze_55814": (["55814 Inel bronz"], "bronze ring", "ring", "flat"),
    "pilum_butt_56378": (["56378 calcai de pilum"], "pilum butt spike", "metal tool", "flat"),
    "spearhead_56385": (["56385 varf de sulita"], "spearhead", "spearhead", "flat"),
    "knife_56394": (["56394 cutit in", "56394 cutit pe"], "knife", "knife", "flat"),
    "knife_curved_56395": (["56395 cutit curb"], "curved knife", "knife", "flat"),
    "razor_56397": (["56397 cutit ras"], "razor knife", "knife", "flat"),
    "bracelet_56439": (["56439 bratara"], "bracelet", "bracelet", "flat"),
    "statuette_bronze_I6138": (["I 6138 statueta bronz"], "bronze statuette", "statuette", "flat"),
    "bust_marble_I7421": (["I 7421 bust marmura"], "marble bust", "sculpture", "upright"),
    "axe_bronze_I25114": (["I25114"], "bronze shaft-hole axe", "axe head", "flat"),
    "axe_bronze_I25132": (["I25132"], "bronze axe", "axe head", "flat"),
    "axe_flat_bronze_I25138": (["I25138"], "flat bronze axe", "axe head", "flat"),
    "axe_stone_perforated_I25675": (["I25675 Topor piatra cu gaura"], "perforated stone axe", "stone axe", "flat"),
    "mould_axe_stone_I25743": (["I25743 Matrita piatra topor"], "stone mould for an axe", "stone", "flat"),
    "jug_I30450": (["I30450"], "jug", "vase", "upright"),
    "amphora_I44639": (["I44639"], "small amphora", "vase", "upright"),
    "jug_I44644": (["I44644"], "jug", "vase", "upright"),
    "hair_ring_silver_I4513": (["I4513 Inel de p"], "silver hair ring", "ring", "flat"),
    "pot_I47499": (["I47499"], "pot", "bowl", "upright"),
    "askos_I50286": (["I50286"], "askos (spouted vessel)", "vase", "upright"),
    "cup_I50287": (["I50287"], "cup", "cup", "upright"),
    "jar_handled_I51410": (["I51410"], "decorated jar with handles", "vase", "upright"),
    "vessel_I56482": (["I56482 Vas ceramic"], "ceramic vessel", "vase", "upright"),
    "cup_I56491": (["I56491"], "cup", "cup", "upright"),
    "pot_I56493": (["I56493"], "small pot", "bowl", "upright"),
    "cup_I56494": (["I56494"], "cup", "cup", "upright"),
    "pot_I56504": (["I56504"], "small pot", "bowl", "upright"),
    "pot_black_I56513": (["I56513"], "small black pot", "bowl", "upright"),
    "beaker_I56515": (["I56515"], "beaker", "vase", "upright"),
    "bowl_I56517": (["I56517"], "bowl", "bowl", "upright"),
    "jug_I7879": (["I7879"], "jug", "vase", "upright"),
    "jug_face_I7928": (["I7928"], "face jug", "vase", "upright"),
    "jug_I7946": (["I7946"], "jug", "vase", "upright"),
    "mug_I8068": (["I8068"], "mug", "cup", "upright"),
    "jug_I8074": (["I8074"], "jug", "cup", "upright"),
    "pot_I8124": (["I8124"], "pot", "bowl", "upright"),
    "vessel_handled_I8224": (["I8224 Vas cu toarta"], "vessel with a handle", "vase", "upright"),
    "jug_I8276": (["I8276 ulcior"], "jug", "vase", "upright"),
    "oven_model_I8285": (["I8285 Cuptor ceramic"], "clay oven model", "pottery", "upright"),
    "altar_small_I8313": (["I8313 altar mic"], "small altar", "pottery", "upright"),
    "cup_two_handled_I9119": (["I9119"], "two-handled cup", "cup", "upright"),
    "vessel_incised_III6415": (["III 6415 vas ceramic incizat"], "incised ceramic vessel", "vase", "upright"),
    "idol_I4583": (["Idol I 4583"], "idol", "figurine", "upright"),
    "idol_I856": (["Idol I 856"], "idol", "figurine", "upright"),
    "idol_dog_I7522": (["Idol caine 7522"], "dog-shaped idol", "figurine", "upright"),
    "idol_bull_I26022": (["Idol vaca I 26022"], "bull-shaped idol", "figurine", "upright"),
    "vessel_I29601": (["I29601 vas ceramic"], "ceramic vessel", "vase", "upright"),
    "spearhead_iron_I40220": (["I40220 varf sulita fier"], "iron spearhead", "spearhead", "flat"),
    "bowl_I40808": (["I40808 castron"], "bowl", "bowl", "upright"),
    "spearhead_iron_I42460": (["I42460 varf sulita fier"], "iron spearhead", "spearhead", "flat"),
    "nail_iron_I44667": (["I44667 cui de fier"], "iron nail", "nail", "flat"),
    "sword_fragment_iron_I44704": (["I44704 fragment sabie fier"], "iron sword fragment", "sword", "flat"),
    "sickle_iron_I4538": (["I4538 secera fier"], "iron sickle", "sickle", "flat"),
    "mould_axe_bivalve_I54478": (["I54478 matri"], "half of a bivalve axe mould", "stone", "flat"),
    "vessel_I55830": (["I55830 vas ceramic"], "ceramic vessel", "vase", "upright"),
    "mug_I55832": (["I55832 canuta ceramica"], "ceramic mug", "cup", "upright"),
    "spearhead_I56410": (["I56410 varf de sulita"], "spearhead", "spearhead", "flat"),
    "spearhead_I56422": (["I56422 varf de sulita"], "spearhead", "spearhead", "flat"),
    "bracelet_spiral_bronze_I56435": (["I56435 bratara bronz plurispiralica"], "spiral bronze bracelet", "bracelet", "flat"),
    "bracelet_bronze_I56436": (["I56436 bratara bronz in", "I56436 bratara bronz pe"], "bronze bracelet", "bracelet", "flat"),
    "sword_iron_I7713": (["I7713 spada fier"], "iron sword", "sword", "flat"),
    "strainer_I7769": (["I7769 obiect ceramic"], "perforated ceramic object (strainer)", "pottery", "upright"),
    "loom_weight_I7806": (["I7806 greutate lut"], "clay loom weight", "stone", "upright"),
    "vessel_I8081": (["I8081 vas ceramic in", "I8081 vas ceramic pe"], "ceramic vessel", "vase", "upright"),
    "vessel_small_I8188": (["I8188 vas ceramic mic"], "small ceramic vessel", "bowl", "upright"),
    "clay_box_M38_2": (["Piesa M 38-2"], "decorated clay box", "pottery", "upright"),
    "pot_18298": (["Vas 18298"], "pot", "bowl", "upright"),
    "cup_I56873": (["Vas 56873"], "Roman cup", "cup", "upright"),
    "bowl_I29667": (["Vas I 29667"], "bowl", "bowl", "upright"),
    "mug_I47602": (["Vas I 47602"], "mug", "cup", "upright"),
    "bowl_I56651": (["Vas I 56651"], "bowl", "bowl", "upright"),
    "cup_I56404": (["Vas I56404"], "cup", "cup", "upright"),
    "jug_M_CXII_431": (["Vas M CXII - 431"], "jug", "vase", "upright"),
    "pot_M_CXII_432": (["Vas M CXII - 432"], "pot", "vase", "upright"),
    "dish_flat": (["Vas plat"], "flat dish", "bowl", "upright"),
    "vessel_incised_X5": (["X5 vas ceramic incizat"], "incised ceramic vessel", "vase", "upright"),
    "bowl_I46785": (["i46785"], "bowl with white inlay", "bowl", "upright"),
    "cup_I56492": (["i56492"], "cup", "cup", "upright"),
}
# heights from the museum's tables (metres)
KNOWN_HEIGHT = {"cup_I56404": 0.097, "bowl_I56651": 0.085}


def main() -> None:
    leaves = sorted({p.parent for p in ROOT.rglob("*.jpg")})
    rows, unmatched = [], []
    for leaf in leaves:
        name = leaf.name
        hits = [k for k, (prefixes, *_) in OBJECTS.items() if any(name == p or name.startswith(p) for p in prefixes)]
        if len(hits) != 1:
            unmatched.append((name, hits))
            continue
        key = hits[0]
        setting = ("lab" if ("laborator" in name or "laboraror" in name)
                   else "white" if ("pe alb" in name or "fond alb" in name) else "mixed")
        _, label, prompt, shape = OBJECTS[key]
        rows.append({"folder": str(leaf), "object": key, "label": label, "setting": setting, "prompt": prompt,
                     "shape": shape, "known_height_m": KNOWN_HEIGHT.get(key), "photos": len(list(leaf.glob("*.jpg")))})
    print("unmatched or ambiguous:", unmatched)
    print(len(rows), "series,", len({r["object"] for r in rows}), "objects,", sum(r["photos"] for r in rows), "photos")
    print("objects without series:", set(OBJECTS) - {r["object"] for r in rows})
    with open(WORK / "series.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
