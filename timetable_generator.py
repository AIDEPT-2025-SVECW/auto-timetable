# File: generate_timetable.py
"""
Production Timetable Generator (weekly-template + remainder placement)
Implements requirements from `automated_timetable_requirements.docx` and the user's requested changes.

Modes:
- debug: tiny satisfiable starter (fast) + verbose variable/constraint counts
- strict: enforce hard constraints (may be infeasible if inputs conflict)
- elastic-debug: allow soft relaxations (penalties) to help solver find feasible solutions

Pipeline overview:
1. Load inputs and precompute mappings (sections→classrooms, subject metadata, eligible faculty - normalized match).
2. Build a compact CP-SAT weekly-template model (day-of-week × period template) that repeats across semester weeks.
   - Weekly template ensures global conflict-free baseline scheduling and enforces most hard constraints.
3. Solve weekly model (with time-limit).
4. Post-process to place remainder session counts (when total required periods don't divide evenly across weeks).
   - Greedy placement across calendar weeks while respecting faculty/section/room availability.
5. Expand final schedule to actual calendar dates and write outputs (section/faculty/classroom JSONs).

Notes and design choices:
- To keep the model tractable we use a weekly template rather than variables per calendar date. Exact total-hours coverage is achieved by:
  - enforcing a weekly minimum frequency in the CP model (floor(required/weeks_count)), and
  - placing remaining sessions greedily across real calendar weeks after solving.
- Faculty eligibility matching ignores case and non-alphanumeric characters.
- Logging is verbose; check `timetable.log` for debugging details.

Run:
    python generate_timetable.py --input-dir ./input --output-dir ./output --mode strict --time-limit 120

"""

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict, Counter
from datetime import datetime

from ortools.sat.python import cp_model

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("timetable.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("timetable")

# ---------------------- Utilities ----------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^0-9a-z]", "", s.lower())


def periods_from_hours(hours: float) -> int:
    # convert hours to 50-minute periods
    minutes = int(round(hours * 60))
    periods = int(round(minutes / 50.0))
    return max(1, periods)


def chunk_weeks(dates):
    # group iso-week -> list of date strings
    dt_objs = [datetime.fromisoformat(d).date() for d in dates]
    dt_objs.sort()
    weeks = defaultdict(list)
    for d in dt_objs:
        iso = d.isocalendar()[:2]
        weeks[iso].append(d.isoformat())
    return list(weeks.values())

# ---------------------- Precomputations ----------------------

def map_sections_to_classrooms(sections, classrooms):
    mapping = {}
    used_room_ids = set()
    available_rooms = [c for c in classrooms if c.get("type") in ("classroom", "lab") and c.get("status") == "active"]
    available_rooms.sort(key=lambda r: r.get("capacity", 0))
    for sec in sections:
        assigned = None
        for room in available_rooms:
            if room["id"] in used_room_ids:
                continue
            if sec.get("totalStudents", 0) <= room.get("capacity", 0):
                assigned = room["id"]
                used_room_ids.add(room["id"])
                break
        if assigned is None and available_rooms:
            assigned = available_rooms[0]["id"]
        mapping[sec["id"]] = assigned
    logger.info("Section->Room mapping computed: assigned %d/%d", sum(1 for v in mapping.values() if v), len(sections))
    return mapping


def build_subjects_meta(subjects):
    meta = {}
    for subj in subjects:
        sid = subj.get("subjectId")
        meta[sid] = {
            "name": subj.get("subjectName"),
            "totalHours": subj.get("totalHours", 0),
            "theoryHours": subj.get("theoryHours", 0),
            "practicalHours": subj.get("practicalHours", 0),
            "isElective": subj.get("isElective", False),
            "electiveGroup": subj.get("elective_group_name")
        }
    return meta


def build_eligible_faculty_map(subjects_meta, faculty_data):
    eligible = {}
    # Build normalized faculty subject lists
    faculty_subjects_norm = {}
    for f in faculty_data:
        fid = f.get("facultyId")
        normalized = {normalize_name(s) for s in f.get("subjects", []) if isinstance(s, str)}
        faculty_subjects_norm[fid] = normalized

    for sid, sm in subjects_meta.items():
        name_norm = normalize_name(sm["name"])
        elig = []
        for f in faculty_data:
            if name_norm in faculty_subjects_norm.get(f.get("facultyId"), set()):
                elig.append(f.get("facultyId"))
        eligible[sid] = elig
        logger.debug("Subject %s eligible faculty count=%d", sm["name"], len(elig))
    return eligible

# ---------------------- Model Builder ----------------------

def build_weekly_model(sections, subjects_meta, faculty_data, weekdays, weeks_count, section_room_map, eligible_map, mode="strict"):
    model = cp_model.CpModel()
    periods_per_day = 8

    section_ids = [s["id"] for s in sections]
    subject_ids = list(subjects_meta.keys())
    faculty_ids = [f.get("facultyId") for f in faculty_data]
    faculty_by_id = {f.get("facultyId"): f for f in faculty_data}

    # precompute
    req_periods = {s: periods_from_hours(subjects_meta[s]["totalHours"]) for s in subject_ids}
    is_lab = {s: (subjects_meta[s]["practicalHours"] and subjects_meta[s]["practicalHours"] > 0) for s in subject_ids}

    # variables
    y = {}  # weekly template slots
    for sec in section_ids:
        for subj in subject_ids:
            for dow in weekdays:
                for p in range(1, periods_per_day + 1):
                    if is_lab[subj] and p == periods_per_day:
                        continue
                    y[(sec, subj, dow, p)] = model.NewBoolVar(f"y_{sec}_{subj}_{dow}_{p}")

    # faculty assignment per subject per section
    assign_fac = {}
    for sec in section_ids:
        for subj in subject_ids:
            elig = eligible_map.get(subj, [])
            if not elig:
                # No eligible faculty -> infeasible
                logger.error("No eligible faculty for subject %s (id=%s). Aborting model build.", subjects_meta[subj]["name"], subj)
                model.Add(0 == 1)
                continue
            for f in elig:
                assign_fac[(sec, subj, f)] = model.NewBoolVar(f"assign_{sec}_{subj}_{f}")
            model.Add(sum(assign_fac[(sec, subj, f)] for f in elig) == 1)

    # z variables linking assignment to weekly occurrences
    z = {}
    for sec in section_ids:
        for subj in subject_ids:
            elig = eligible_map.get(subj, [])
            for f in elig:
                for dow in weekdays:
                    for p in range(1, periods_per_day + 1):
                        if is_lab[subj] and p == periods_per_day:
                            continue
                        yvar = y[(sec, subj, dow, p)]
                        af = assign_fac[(sec, subj, f)]
                        zv = model.NewBoolVar(f"z_{sec}_{subj}_{f}_{dow}_{p}")
                        z[(sec, subj, f, dow, p)] = zv
                        model.Add(zv <= yvar)
                        model.Add(zv <= af)
                        model.Add(zv >= yvar + af - 1)

    # hard constraints
    # 1) Section: at most 1 subject per (dow,period), at least one free period (reserve 1)
    for sec in section_ids:
        for dow in weekdays:
            occ = []
            for subj in subject_ids:
                for p in range(1, periods_per_day + 1):
                    if (sec, subj, dow, p) in y:
                        occ.append(y[(sec, subj, dow, p)])
            model.Add(sum(occ) <= periods_per_day - 1)

    # 2) same theory subject not more than once per day
    for sec in section_ids:
        for subj in subject_ids:
            if is_lab[subj]:
                continue
            for dow in weekdays:
                slots = [y[(sec, subj, dow, p)] for p in range(1, periods_per_day + 1) if (sec, subj, dow, p) in y]
                if slots:
                    model.Add(sum(slots) <= 1)

    # 3) lab sessions: at least 3 lab periods per week per section (each lab start counts as 2)
    for sec in section_ids:
        lab_subjs = [s for s in subject_ids if is_lab[s]]
        if not lab_subjs:
            continue
        lab_terms = []
        for subj in lab_subjs:
            for dow in weekdays:
                for p in range(1, periods_per_day):
                    if (sec, subj, dow, p) in y:
                        lab_terms.append(y[(sec, subj, dow, p)] * 2)
        if lab_terms:
            model.Add(sum(lab_terms) >= 3)

    # 4) faculty: no double booking per dow & period
    for f in faculty_ids:
        for dow in weekdays:
            for p in range(1, periods_per_day + 1):
                occ = []
                for sec in section_ids:
                    for subj in subject_ids:
                        key = (sec, subj, f, dow, p)
                        if key in z:
                            occ.append(z[key])
                if occ:
                    model.Add(sum(occ) <= 1)

    # 5) faculty daily maxima hard, minima depends on mode
    for f in faculty_ids:
        for dow in weekdays:
            occ_terms = []
            for sec in section_ids:
                for subj in subject_ids:
                    for p in range(1, periods_per_day + 1):
                        key = (sec, subj, f, dow, p)
                        if key in z:
                            if is_lab[subj]:
                                occ_terms.append(z[key] * 2)
                            else:
                                occ_terms.append(z[key])
            if occ_terms:
                model.Add(sum(occ_terms) <= 4)
                if mode == "strict":
                    model.Add(sum(occ_terms) >= 2)
                elif mode == "elastic-debug":
                    # add soft lower bound via slack variable
                    slack = model.NewIntVar(0, 8, f"slack_min_{f}_{dow}")
                    model.Add(sum(occ_terms) + slack >= 2)
                    # will minimize slack in objective later

    # 6) required weekly minimums: compute weekly_min = floor(total/weeks_count)
    weekly_min = {}
    remainders = {}
    for sec in section_ids:
        for subj in subject_ids:
            total_req = req_periods[subj]
            wm = total_req // weeks_count
            r = total_req - wm * weeks_count
            weekly_min[(sec, subj)] = wm
            remainders[(sec, subj)] = r
            # sum_weekly >= wm
            terms = []
            for dow in weekdays:
                for p in range(1, periods_per_day + 1):
                    if (sec, subj, dow, p) in y:
                        if is_lab[subj]:
                            terms.append(y[(sec, subj, dow, p)] * 2)
                        else:
                            terms.append(y[(sec, subj, dow, p)])
            if terms:
                model.Add(sum(terms) >= wm)
            else:
                if wm > 0:
                    model.Add(0 == wm)  # force infeasible to show diagnostics

    # Soft constraints/objective
    penalties = []
    penalty_weights = []

    # avoid consecutive theory classes per section per dow (soft)
    for sec in section_ids:
        for dow in weekdays:
            for p in range(1, periods_per_day):
                theory_p = []
                theory_p1 = []
                for subj in subject_ids:
                    if is_lab[subj]:
                        continue
                    if (sec, subj, dow, p) in y:
                        theory_p.append(y[(sec, subj, dow, p)])
                    if (sec, subj, dow, p + 1) in y:
                        theory_p1.append(y[(sec, subj, dow, p + 1)])
                if theory_p and theory_p1:
                    c = model.NewBoolVar(f"consec_{sec}_{dow}_{p}")
                    # c can be 1 only if both sides occupied
                    # sum(theory_p) + sum(theory_p1) - c <= len(theory_p) + len(theory_p1) - 1
                    model.Add(sum(theory_p) + sum(theory_p1) - c <= len(theory_p) + len(theory_p1) - 1)
                    penalties.append(c)
                    penalty_weights.append(1)

    # If elastic-debug, add slack variables for other soft constraints as needed (e.g., faculty min slack collected earlier)

    # Build objective
    if penalties:
        if mode == "elastic-debug":
            # minimize penalties with weight
            obj_terms = []
            for i, pvar in enumerate(penalties):
                obj_terms.append(pvar * penalty_weights[i])
            model.Minimize(sum(obj_terms))
        else:
            model.Minimize(sum(penalties))

        logger.info("Built weekly model (sections=%d subjects=%d faculties=%d weeks=%d weekdays=%s)",
            len(section_ids), len(subject_ids), len(faculty_ids), weeks_count, weekdays)

    meta = {
        "y": y,
        "assign_fac": assign_fac,
        "z": z,
        "section_ids": section_ids,
        "subject_ids": subject_ids,
        "faculty_ids": faculty_ids,
        "faculty_by_id": faculty_by_id,
        "weekdays": weekdays,
        "periods_per_day": periods_per_day,
        "is_lab": is_lab,
        "subjects_meta": subjects_meta,
        "section_room_map": section_room_map,
        "eligible_map": eligible_map,
        "weekly_min": weekly_min,
        "remainders": remainders,
        "weeks_count": weeks_count,
    }
    return model, meta

# ---------------------- Postprocess: place remainders greedily ----------------------

def place_remainders_greedy(weeks_list, meta, weekly_solution, assign_fac_chosen):
    """
    weeks_list: list of weeks where each week is list of date strings
    meta: model metadata
    weekly_solution: dict mapping (sec,subj,dow,p)->bool from weekly template
    assign_fac_chosen: dict mapping (sec,subj)->facultyId from solver
    """
    section_ids = meta["section_ids"]
    subject_ids = meta["subject_ids"]
    is_lab = meta["is_lab"]
    periods_per_day = meta["periods_per_day"]
    weekdays = meta["weekdays"]

    # Build occupancy for each week index: section->(dow,p) occupied by weekly template
    weeks_occupancy = []
    for wk_idx, wk_dates in enumerate(weeks_list):
        occ = {sec: [[False] * (periods_per_day + 1) for _ in range(8)] for sec in section_ids}
        # initialize with weekly template: for each dow in week, find date with that dow
        for date_str in wk_dates:
            dt = datetime.fromisoformat(date_str).date()
            dow = dt.isoweekday()
            if dow not in weekdays:
                continue
            for sec in section_ids:
                for subj in subject_ids:
                    for p in range(1, periods_per_day + 1):
                        key = (sec, subj, dow, p)
                        if weekly_solution.get(key, False):
                            occ[sec][dow][p] = True
                            if is_lab[subj] and p + 1 <= periods_per_day:
                                occ[sec][dow][p + 1] = True
        weeks_occupancy.append(occ)

    # Faculty occupancy per week,dow,period
    faculty_week_occ = []
    for wk_idx, wk_dates in enumerate(weeks_list):
        focc = defaultdict(lambda: defaultdict(lambda: [False] * (periods_per_day + 1)))
        # populate from weekly_solution and assign_fac_chosen
        for date_str in wk_dates:
            dt = datetime.fromisoformat(date_str).date()
            dow = dt.isoweekday()
            if dow not in weekdays:
                continue
            for sec in section_ids:
                for subj in subject_ids:
                    fac = assign_fac_chosen.get((sec, subj))
                    for p in range(1, periods_per_day + 1):
                        if weekly_solution.get((sec, subj, dow, p), False):
                            if fac:
                                focc[fac][dow][p] = True
                                if is_lab[subj] and p + 1 <= periods_per_day:
                                    focc[fac][dow][p + 1] = True
        faculty_week_occ.append(focc)

    # Now greedy allocate remainders
    placements = []  # tuples (week_idx, sec, subj, dow, p)
    unscheduled = []
    for sec in section_ids:
        for subj in subject_ids:
            r = meta["remainders"].get((sec, subj), 0)
            if not r:
                continue
            fac = assign_fac_chosen.get((sec, subj))
            if not fac:
                unscheduled.append((sec, subj, r, "no_assigned_fac"))
                continue
            placed = 0
            # try each week and each slot
            for wk_idx, wk_dates in enumerate(weeks_list):
                if placed >= r:
                    break
                for date_str in wk_dates:
                    if placed >= r:
                        break
                    dt = datetime.fromisoformat(date_str).date()
                    dow = dt.isoweekday()
                    if dow not in weekdays:
                        continue
                    for p in range(1, periods_per_day + 1):
                        if is_lab[subj] and p == periods_per_day:
                            continue
                        # check section free
                        if weeks_occupancy[wk_idx][sec][dow][p]:
                            continue
                        # check subsequent period for lab
                        if is_lab[subj] and weeks_occupancy[wk_idx][sec][dow][p + 1]:
                            continue
                        # check faculty free
                        if faculty_week_occ[wk_idx][fac][dow][p]:
                            continue
                        if is_lab[subj] and faculty_week_occ[wk_idx][fac][dow][p + 1]:
                            continue
                        # assign
                        weeks_occupancy[wk_idx][sec][dow][p] = True
                        if is_lab[subj] and p + 1 <= periods_per_day:
                            weeks_occupancy[wk_idx][sec][dow][p + 1] = True
                        faculty_week_occ[wk_idx][fac][dow][p] = True
                        if is_lab[subj] and p + 1 <= periods_per_day:
                            faculty_week_occ[wk_idx][fac][dow][p + 1] = True
                        placements.append((wk_idx, sec, subj, dow, p))
                        placed += 1
                        if placed >= r:
                            break
            if placed < r:
                unscheduled.append((sec, subj, r - placed, "remainder_unplaced"))

    return placements, unscheduled

# ---------------------- Extract & Write Final Timetable ----------------------

def expand_and_write(weekly_solution, placements, weeks_list, meta, assign_fac_chosen, output_dir):
    # build per-date final schedule by expanding weekly_solution and adding placements
    section_ids = meta["section_ids"]
    subject_ids = meta["subject_ids"]
    is_lab = meta["is_lab"]
    periods_per_day = meta["periods_per_day"]
    weekdays = meta["weekdays"]
    subjects_meta = meta["subjects_meta"]
    section_room_map = meta["section_room_map"]
    faculty_by_id = meta["faculty_by_id"]

    # initialize outputs
    section_timetables = {s: [] for s in section_ids}
    faculty_timetables = defaultdict(list)
    classroom_timetables = defaultdict(list)

    # convert placements into a lookup per week
    placement_lookup = defaultdict(list)
    for wk_idx, sec, subj, dow, p in placements:
        placement_lookup[wk_idx].append((sec, subj, dow, p))

    for wk_idx, wk_dates in enumerate(weeks_list):
        for date_str in wk_dates:
            dt = datetime.fromisoformat(date_str).date()
            dow = dt.isoweekday()
            if dow not in weekdays:
                continue
            for sec in section_ids:
                occupied = [False] * (periods_per_day + 1)
                # weekly solution
                for subj in subject_ids:
                    for p in range(1, periods_per_day + 1):
                        if weekly_solution.get((sec, subj, dow, p), False):
                            fac = assign_fac_chosen.get((sec, subj))
                            subj_name = subjects_meta[subj]["name"]
                            entry = {
                                "section": sec,
                                "date": date_str,
                                "period": p,
                                "subjectId": subj,
                                "subjectName": subj_name,
                                "facultyId": fac,
                                "facultyName": faculty_by_id.get(fac, {}).get("name"),
                                "room": section_room_map.get(sec),
                                "isLabStart": is_lab[subj],
                                "isElective": subjects_meta[subj].get("isElective", False),
                            }
                            section_timetables[sec].append(entry)
                            faculty_timetables[fac].append(entry)
                            classroom_timetables[section_room_map.get(sec)].append(entry)
                            occupied[p] = True
                            if is_lab[subj] and p + 1 <= periods_per_day:
                                occupied[p + 1] = True
                # placements for this week
                for (sec_p, subj_p, dow_p, p_p) in placement_lookup.get(wk_idx, []):
                    if sec_p != sec:
                        continue
                    if dow_p != dow:
                        continue
                    fac = assign_fac_chosen.get((sec_p, subj_p))
                    subj_name = subjects_meta[subj_p]["name"]
                    entry = {
                        "section": sec_p,
                        "date": date_str,
                        "period": p_p,
                        "subjectId": subj_p,
                        "subjectName": subj_name,
                        "facultyId": fac,
                        "facultyName": faculty_by_id.get(fac, {}).get("name"),
                        "room": section_room_map.get(sec_p),
                        "isLabStart": is_lab[subj_p],
                        "isElective": subjects_meta[subj_p].get("isElective", False),
                    }
                    section_timetables[sec].append(entry)
                    faculty_timetables[fac].append(entry)
                    classroom_timetables[section_room_map.get(sec)].append(entry)
                    occupied[p_p] = True
                    if is_lab[subj_p] and p_p + 1 <= periods_per_day:
                        occupied[p_p + 1] = True

                # fill free periods
                for p in range(1, periods_per_day + 1):
                    if not occupied[p]:
                        section_timetables[sec].append({
                            "section": sec,
                            "date": date_str,
                            "period": p,
                            "free": True
                        })

    # write outputs
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "section_timetables.json"), "w", encoding="utf-8") as f:
        json.dump(section_timetables, f, indent=2)
    with open(os.path.join(output_dir, "faculty_timetables.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in faculty_timetables.items()}, f, indent=2)
    with open(os.path.join(output_dir, "classroom_timetables.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in classroom_timetables.items()}, f, indent=2)

    logger.info("Final timetables written to %s", output_dir)

# ---------------------- Debug Mode Builder ----------------------

def build_debug_model(sections, subjects, faculty, weekdays):
    """Small model for quick validation"""
    model = cp_model.CpModel()
    periods_per_day = 4
    section_ids = [s["id"] for s in sections[:1]]
    subject_ids = [subj.get("subjectId") for subj in subjects[:2]]
    faculty_ids = [f.get("facultyId") for f in faculty[:2]]

    y = {}
    for sec in section_ids:
        for subj in subject_ids:
            for dow in weekdays[:3]:
                for p in range(1, periods_per_day + 1):
                    y[(sec, subj, dow, p)] = model.NewBoolVar(f"y_{sec}_{subj}_{dow}_{p}")

    # simple: each period at most 1
    for sec in section_ids:
        for dow in weekdays[:3]:
            for p in range(1, periods_per_day + 1):
                model.Add(sum(y[(sec, subj, dow, p)] for subj in subject_ids) <= 1)

    # each subject must appear at least once in the week
    for subj in subject_ids:
        model.Add(sum(y[(sec, subj, dow, p)] for sec in section_ids for dow in weekdays[:3] for p in range(1, periods_per_day + 1)) >= 1)

        logger.info("Debug model built (sections=%d subjects=%d faculties=%d weekdays=%s)",
                len(section_ids), len(subject_ids), len(faculty_ids), weekdays[:3])
    return model, y

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=False, default="./input")
    parser.add_argument("--output-dir", required=False, default="./output")
    parser.add_argument("--mode", choices=["strict", "elastic-debug", "debug"], default="elastic-debug")
    parser.add_argument("--time-limit", type=int, default=120)
    args = parser.parse_args()

    try:
        faculty_data = load_json(os.path.join(args.input_dir, "aiml-faculty-detailed.json"))
        semester_dates = load_json(os.path.join(args.input_dir, "all_semesters_net_dates.json"))
        classrooms = load_json(os.path.join(args.input_dir, "classrooms.json"))
        sections = load_json(os.path.join(args.input_dir, "department-sections.json"))
        subjects = load_json(os.path.join(args.input_dir, "semester_subjects.json"))
    except Exception as e:
        logger.exception("Failed to load inputs: %s", e)
        return

    logger.info("Inputs: sections=%d subjects=%d faculty=%d classrooms=%d semesters=%d",
                len(sections), len(subjects), len(faculty_data), len(classrooms), len(semester_dates))

    # quick debug mode
    all_dates = []
    for sem in semester_dates:
        all_dates.extend(sem.get("netDates", []))
    all_dates = sorted(list(set(all_dates)))
    if not all_dates:
        logger.error("No net dates found. Exiting.")
        return

    weeks = chunk_weeks(all_dates)
    weeks_count = len(weeks)
    weekday_set = sorted({datetime.fromisoformat(d).isoweekday() for d in all_dates})

    logger.info("Calendar: weeks=%d weekday_set=%s", weeks_count, weekday_set)

    if args.mode == "debug":
        model, y = build_debug_model(sections, subjects, faculty_data, weekday_set)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.time_limit
        status = solver.Solve(model)
        logger.info("Debug solve status=%s", solver.StatusName(status));
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            # print some y variables
            cnt = 0
            for k, v in y.items():
                if solver.Value(v):
                    logger.info("Scheduled (debug): %s", k)
                    cnt += 1
                    if cnt >= 20:
                        break
        return

    # production modes
    section_room_map = map_sections_to_classrooms(sections, classrooms)
    subjects_meta = build_subjects_meta(subjects)
    eligible_map = build_eligible_faculty_map(subjects_meta, faculty_data)

    model, meta = build_weekly_model(sections, subjects_meta, faculty_data, weekday_set, weeks_count, section_room_map, eligible_map, args.mode)

    logger.info("Model summary (sections=%d subjects=%d faculties=%d weeks=%d weekdays=%s)",
            len(meta["section_ids"]), len(meta["subject_ids"]), len(meta["faculty_ids"]), meta["weeks_count"], meta["weekdays"])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    solver.parameters.num_search_workers = 8

    logger.info("Solving model (mode=%s time_limit=%ds)...", args.mode, args.time_limit)
    status = solver.Solve(model)
    logger.info("Solver finished: %s", solver.StatusName(status))

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        logger.error("No feasible weekly-template solution. See timetable.log for details.")
        ensure_dir(args.output_dir)
        with open(os.path.join(args.output_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump({
            "status": solver.StatusName(status),
            "sections": len(meta["section_ids"]),
            "subjects": len(meta["subject_ids"]),
            "faculties": len(meta["faculty_ids"]),
             "weeks": meta["weeks_count"],
            "weekdays": meta["weekdays"]
            }, f, indent=2)
        return

    # extract weekly solution
    weekly_solution = {}
    for (sec, subj, dow, p), var in meta["y"].items():
        val = solver.Value(var)
        weekly_solution[(sec, subj, dow, p)] = bool(val)

    # extract chosen faculty per (sec,subj)
    assign_fac_chosen = {}
    for (sec, subj, f), var in meta["assign_fac"].items():
        if solver.Value(var):
            assign_fac_chosen[(sec, subj)] = f

    logger.info("Weekly template extracted. Chosen faculty assignments: %d", len(assign_fac_chosen))

    # place remainders greedily
    placements, unscheduled = place_remainders_greedy(weeks, meta, weekly_solution, assign_fac_chosen)

    logger.info("Placed %d remainder sessions; unscheduled_remainders=%d", len(placements), len(unscheduled))
    if unscheduled:
        logger.warning("Unscheduled remainders sample: %s", unscheduled[:20])

    # expand to dates and write outputs
    expand_and_write(weekly_solution, placements, weeks, meta, assign_fac_chosen, args.output_dir)

    # diagnostics
    ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, "run_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "solver_status": solver.StatusName(status),
            "sections": len(meta["section_ids"]),
            "subjects": len(meta["subject_ids"]),
            "faculties": len(meta["faculty_ids"]),
            "weeks": meta["weeks_count"],
            "weekdays": meta["weekdays"],
            "placements": len(placements),
            "unscheduled_remainders": unscheduled,
        }, f,  indent=2)

    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
