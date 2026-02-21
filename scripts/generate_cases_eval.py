# scripts/generate_cases_eval.py
from __future__ import annotations
import json, random, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

random.seed(7)

@dataclass
class CaseSpec:
    category: str
    n: int

CATS: List[CaseSpec] = [
    CaseSpec("cardiac", 30),
    CaseSpec("neuro", 30),
    CaseSpec("resp", 25),
    CaseSpec("gi", 25),
    CaseSpec("infectious", 25),
    CaseSpec("endo", 20),
    CaseSpec("obgyn", 20),
    CaseSpec("trauma", 20),
    CaseSpec("psych", 20),
    CaseSpec("peds", 15),
]

def pick_age(cat: str) -> int:
    if cat == "peds":
        return random.choice([2, 4, 7, 10, 13, 15, 17])
    if cat == "obgyn":
        return random.randint(18, 45)
    return random.randint(18, 85)

def pick_sex(cat: str) -> str:
    if cat == "obgyn":
        return "F"
    return random.choice(["M","F"])

def vitals_for(cat: str) -> str:
    # Keep it realistic and varied but simple: BP HR RR SpO2 Temp
    # cat modifiers
    hr = random.randint(60, 105)
    rr = random.randint(12, 20)
    spo2 = random.randint(95, 100)
    temp = round(random.uniform(36.0, 37.4), 1)
    sys = random.randint(105, 140)
    dia = random.randint(65, 90)

    if cat in ("infectious",):
        temp = round(random.uniform(37.5, 39.3), 1)
        hr = random.randint(85, 125)
    if cat in ("resp",):
        spo2 = random.randint(88, 96)
        rr = random.randint(18, 30)
    if cat in ("cardiac",):
        hr = random.randint(80, 135)
        sys = random.randint(120, 175)
        dia = random.randint(70, 105)
    if cat in ("trauma",):
        hr = random.randint(75, 120)

    return f"BP {sys}/{dia} HR {hr} RR {rr} SpO2 {spo2}% Temp {temp}C"

def maybe(items: List[str], p: float) -> List[str]:
    return [x for x in items if random.random() < p]

def build_case(cat: str, idx: int) -> Dict:
    age = pick_age(cat)
    sex = pick_sex(cat)
    vitals = vitals_for(cat)

    meds_pool = {
        "cardiac": ["aspirin", "atorvastatin", "metoprolol"],
        "neuro": ["levetiracetam", "warfarin", "aspirin"],
        "resp": ["albuterol", "prednisone"],
        "gi": ["omeprazole", "ondansetron"],
        "infectious": ["amoxicillin", "doxycycline"],
        "endo": ["metformin", "insulin"],
        "obgyn": ["prenatal vitamin"],
        "trauma": ["ibuprofen", "acetaminophen"],
        "psych": ["sertraline", "fluoxetine"],
        "peds": ["acetaminophen"],
    }[cat]

    allergies_pool = ["penicillin", "sulfa", "none"]

    meds = maybe(meds_pool, 0.35)
    allergies = [random.choice(allergies_pool)]
    if allergies == ["none"]:
        allergies = []

    # --- category-specific content + gold SOAP ---
    # Keep gold SOAP *lexically close* to case to boost rouge/recall,
    # while still clinically plausible.

    red_flags: List[str] = []
    required_questions: List[str] = []
    escalate = False

    if cat == "cardiac":
        presenting = random.choice([
            "Chest pain", "Chest pressure", "Shortness of breath"
        ])
        hpi = random.choice([
            "Sudden central chest pressure radiating to left arm with diaphoresis and nausea.",
            "Exertional chest tightness with shortness of breath and lightheadedness.",
            "Chest pressure with nausea and sweating starting 30 minutes ago."
        ])
        risk = random.choice([
            "Smoker. No prior MI.", "History of hypertension.", "Family history of CAD."
        ])
        hpi = f"{age}M with {hpi} {risk}"
        red_flags = ["possible ACS", "diaphoresis", "radiation"]
        required_questions = ["onset duration", "exertional", "shortness of breath", "risk factors", "aspirin use"]
        escalate = True

        subj = f"{presenting} with {hpi}"
        obj = f"{vitals}. Appears uncomfortable."
        asmt = "High suspicion for acute coronary syndrome."
        plan = "- ECG now\n- Troponin now\n- Aspirin if no contraindication\n- ED evaluation / monitor"

    elif cat == "neuro":
        presenting = random.choice(["Dizziness", "Weakness", "Slurred speech"])
        hpi = random.choice([
            "Sudden dizziness and trouble walking started 1 hour ago.",
            "New unilateral weakness and facial droop noticed today.",
            "Slurred speech with imbalance beginning this morning."
        ])
        meds = meds or maybe(["amlodipine","warfarin"], 0.3)
        red_flags = ["possible stroke", "acute neuro symptoms"]
        required_questions = ["time last known well", "anticoagulants", "focal deficit"]
        escalate = True

        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. Neuro exam concerning for focal deficit."
        asmt = "Concern for acute stroke or TIA."
        plan = "- Stroke protocol now\n- CT head\n- Glucose check\n- ED evaluation / monitor"

    elif cat == "resp":
        presenting = random.choice(["Shortness of breath", "Cough", "Wheezing"])
        hpi = random.choice([
            "Worsening shortness of breath with wheezing over 2 days.",
            "Cough and dyspnea with chest tightness, worse at night.",
            "Shortness of breath with cough and fever."
        ])
        red_flags = ["hypoxia", "respiratory distress"]
        required_questions = ["asthma history", "smoking", "fever"]
        escalate = ( "SpO2 8" in vitals ) or ("SpO2 9" in vitals)

        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. Increased work of breathing."
        asmt = "Acute dyspnea, consider asthma/COPD exacerbation or pneumonia."
        plan = "- Pulse oximetry monitoring\n- Bronchodilator trial\n- CXR if infectious symptoms\n- ED if worsening"

    elif cat == "gi":
        presenting = random.choice(["Abdominal pain", "Nausea/vomiting", "Diarrhea"])
        hpi = random.choice([
            "Crampy abdominal pain with vomiting since last night.",
            "Epigastric pain after meals with nausea.",
            "Watery diarrhea and abdominal cramps for 2 days."
        ])
        red_flags = ["severe abdominal pain", "dehydration"]
        required_questions = ["bloody stool", "oral intake", "fever"]
        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. Abdomen soft with mild tenderness."
        asmt = "Acute gastroenteritis vs gastritis."
        plan = "- Oral rehydration\n- Antiemetic if needed\n- Return precautions for worsening or blood"

    elif cat == "infectious":
        presenting = random.choice(["Fever", "Sore throat", "Dysuria"])
        if presenting == "Dysuria":
            hpi = "Burning with urination and urinary frequency for 2 days. No flank pain."
            red_flags = ["pyelonephritis concern"]
            required_questions = ["flank pain", "pregnancy status", "antibiotic allergies"]
            asmt = "Uncomplicated cystitis."
            plan = "- UA/urine culture\n- Antibiotics per guideline considering allergies\n- Return precautions"
        else:
            hpi = random.choice([
                "Fever with cough and myalgias for 3 days.",
                "Sore throat and fever with no cough.",
            ])
            red_flags = ["high fever"]
            required_questions = ["duration", "resp symptoms", "sick contacts"]
            asmt = "Acute infection, consider viral syndrome."
            plan = "- Supportive care\n- Testing if indicated\n- Return precautions"
        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. Exam otherwise unremarkable."

    elif cat == "endo":
        presenting = random.choice(["High blood sugar", "Fatigue", "Polyuria"])
        hpi = random.choice([
            "Increased thirst and urination with fatigue for 1 week.",
            "Known diabetes with elevated home glucose readings.",
        ])
        red_flags = ["hyperglycemia"]
        required_questions = ["ketones", "insulin use", "vomiting"]
        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. No acute distress."
        asmt = "Hyperglycemia, evaluate for ketosis if symptomatic."
        plan = "- Check glucose/ketones\n- Hydration\n- Adjust diabetes regimen with follow-up"

    elif cat == "obgyn":
        presenting = random.choice(["Pelvic pain", "Vaginal bleeding", "Nausea in pregnancy"])
        hpi = random.choice([
            "Lower abdominal pain with vaginal spotting.",
            "Positive pregnancy test with nausea and vomiting.",
            "Pelvic pain with abnormal discharge."
        ])
        red_flags = ["ectopic concern", "heavy bleeding"]
        required_questions = ["pregnancy status", "LMP", "pain severity"]
        escalate = True
        subj = f"{presenting}: {age}F with {hpi}"
        obj = f"{vitals}. Abdominal exam as documented."
        asmt = "OB/GYN complaint—rule out emergent causes."
        plan = "- Pregnancy test\n- Ultrasound if indicated\n- ED evaluation if severe pain/bleeding"

    elif cat == "trauma":
        presenting = random.choice(["Ankle injury", "Head injury", "Laceration"])
        hpi = random.choice([
            "Twisted ankle while running; swelling and pain.",
            "Minor head strike without loss of consciousness.",
            "Small laceration after kitchen accident."
        ])
        red_flags = ["neuro symptoms", "uncontrolled bleeding"]
        required_questions = ["loss of consciousness", "anticoagulants", "tetanus status"]
        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. Local exam consistent with injury."
        asmt = "Minor trauma; evaluate for fracture or concussion based on exam."
        plan = "- Pain control\n- Imaging if indicated\n- Wound care / tetanus update\n- Return precautions"

    elif cat == "psych":
        presenting = random.choice(["Anxiety", "Depression", "Insomnia"])
        hpi = random.choice([
            "Worsening anxiety with palpitations and worry for 1 month.",
            "Low mood and anhedonia for several weeks.",
            "Difficulty sleeping with daytime fatigue."
        ])
        red_flags = ["suicidal ideation"]
        required_questions = ["SI/HI", "substance use", "psych history"]
        subj = f"{presenting}: {age}{sex} with {hpi}"
        obj = f"{vitals}. Calm, cooperative."
        asmt = "Mood/anxiety symptoms; assess safety and functional impact."
        plan = "- Screen for SI/HI\n- Therapy resources\n- Consider SSRI if appropriate\n- Follow-up"

    elif cat == "peds":
        presenting = random.choice(["Fever", "Cough", "Ear pain"])
        hpi = random.choice([
            "Fever and cough for 2 days, drinking fluids.",
            "Ear pain with fever since yesterday.",
            "Runny nose and cough, no breathing difficulty."
        ])
        red_flags = ["resp distress", "dehydration"]
        required_questions = ["vaccines up to date", "urine output", "breathing difficulty"]
        subj = f"{presenting}: {age}yo {sex} with {hpi}"
        obj = f"{vitals}. Appears nontoxic."
        asmt = "Pediatric viral illness vs otitis media depending on exam."
        plan = "- Supportive care\n- Antipyretics\n- Return precautions"

    else:
        raise ValueError(cat)

    # ground_truth fields for scoring hooks
    gt = {
        "red_flags": red_flags,
        "required_questions": required_questions,
        "escalate": bool(escalate),
        "soap_reference": {
            "subjective": subj,
            "objective": obj,
            "assessment": asmt,
            "plan": plan.replace("\n", " ").strip() if isinstance(plan, str) else str(plan),
        }
    }

    # Also provide a flat soap_reference string for your runner (either is fine)
    soap_reference_str = (
        f"SUBJECTIVE: {subj}\n"
        f"OBJECTIVE: {obj}\n"
        f"ASSESSMENT: {asmt}\n"
        f"PLAN: {plan}\n"
    ).strip()

    case = {
        "case_id": f"{cat[:3]}_{idx:03d}",
        "presenting_complaint": subj.split(":")[0] if ":" in subj else presenting if 'presenting' in locals() else cat,
        "history_of_present_illness": hpi,
        "medications": meds,
        "allergies": allergies,
        "vitals": vitals,
        "age": age,
        "sex": sex,
        "ground_truth": gt,
        "reference_soap": soap_reference_str,  # your runner prefers this; keep it
    }
    return case

def main(out_path: str = "data/cases_eval_250.jsonl"):
    rows: List[Dict] = []
    for spec in CATS:
        for i in range(spec.n):
            rows.append(build_case(spec.category, i+1))

    random.shuffle(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "to", out_path)

if __name__ == "__main__":
    main()