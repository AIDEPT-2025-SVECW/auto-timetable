#!/usr/bin/env python3
"""
generate_timetable_json.py

Produces:
 - section_timetable.json  (section-wise, day-wise timetable)
 - faculty_allocations.json (faculty-wise allocations and stats)

Inputs (expected in ./input):
 - aiml-faculty-detailed.json
 - semester-subjects.json
 - department-sections.json
 - classrooms.json
 - all_semesters_net_dates.json

Install:
 pip install ortools

Run:
 python generate_timetable_json.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime
from ortools.sat.python import cp_model

# ---------- Config ----------
INPUT_DIR = "input"
FILES = {
    "faculty": "aiml-faculty-detailed.json",
    "subjects": "semester-subjects.json",
    "sections": "department-sections.json",
    "rooms": "classrooms.json",
    "dates": "all_semesters_net_dates.json",
}

PERIODS_PER_DAY = 8
LAB_BLOCK = 2  # lab = 2 consecutive periods
MIN_FACULTY_PER_DAY = 2
MAX_FACULTY_PER_DAY = 5
MIN_LABS_PER_FACULTY_PER_WEEK = 2
MIN_LABS_PER_SECTION_PER_WEEK = 3
FREE_PERIODS_PER_SECTION_PER_DAY = 1
SOLVER_TIME_LIMIT_SECONDS = 60

SEMESTER_ID = "1-1"  # change if needed

# ---------- Helpers ----------
def load(fn):
    path = os.path.join(INPUT_DIR, fn)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iso_week(dstr):
    d = datetime.strptime(dstr, "%Y-%m-%d")
    return d.isocalendar()[0], d.isocalendar()[1]

def iso_month(dstr):
    d = datetime.strptime(dstr, "%Y-%m-%d")
    return (d.year, d.month)

def normalize_key(s): return s.strip().lower()

# ---------- Load inputs ----------
faculty = load(FILES["faculty"])
subjects = load(FILES["subjects"])
sections = load(FILES["sections"])
rooms = load(FILES["rooms"])
all_dates = load(FILES["dates"])

sem_entry = next((s for s in all_dates if s["semesterId"] == SEMESTER_ID), None)
if not sem_entry:
    raise SystemExit(f"Semester {SEMESTER_ID} not found")
DATES = sem_entry["netDates"]

# index rooms
rooms_by_type = defaultdict(list)
for r in rooms:
    rooms_by_type[r["type"]].append(r)

lecture_rooms = rooms_by_type.get("classroom", [])
lab_rooms = rooms_by_type.get("lab", lecture_rooms)

semester_num = int(SEMESTER_ID.split("-")[0])
subjects_for_sem = [s for s in subjects if s.get("semester") == semester_num and s.get("departmentId","").lower()=="aiml"]

sections_to_schedule = [sec for sec in sections if sec.get("year") == semester_num]
if not sections_to_schedule:
    sections_to_schedule = sections

# faculty index
fac_by_subj = defaultdict(list)
for f in faculty:
    for sname in f.get("subjects", []):
        fac_by_subj[normalize_key(sname)].append(f)

# ---------- Build tasks ----------
tasks = []
for subj in subjects_for_sem:
    subj_name = subj["subjectName"]
    th = subj.get("theoryHours", 0)
    pr = subj.get("practicalHours", 0)
    theory_periods = int(round(th * 60.0 / 50.0))
    practical_periods = int(round(pr * 60.0 / 50.0))
    lab_sessions = (practical_periods + LAB_BLOCK - 1) // LAB_BLOCK
    for sec in sections_to_schedule:
        tasks.append({
            "subjectId": subj["subjectId"],
            "subjectName": subj_name,
            "sectionId": sec["id"],
            "sectionName": sec["name"],
            "theory_periods": theory_periods,
            "lab_sessions": lab_sessions,
        })

# ---------- Faculty assignment ----------
task_faculty = {}
unassigned = []
for t in tasks:
    key = normalize_key(t["subjectName"])
    candidates = fac_by_subj.get(key, [])
    if not candidates:
        for k, facs in fac_by_subj.items():
            if key in k or k in key:
                candidates = facs
                break
    if not candidates:
        task_faculty[(t["sectionId"], t["subjectId"])] = None
        unassigned.append((t["sectionId"], t["subjectName"]))
    else:
        chosen = sorted(candidates, key=lambda f: f.get("experience",0))[0]
        task_faculty[(t["sectionId"], t["subjectId"])] = chosen["facultyId"]

if unassigned:
    print("⚠️ Unassigned (section,subject) pairs:", unassigned)

# ---------- Model ----------
model = cp_model.CpModel()
assign_th, assign_lab = {}, {}

# create vars
for t in tasks:
    sid, subj = t["sectionId"], t["subjectId"]
    fid = task_faculty.get((sid, subj))
    if fid is None:
        continue
    for d_idx in range(len(DATES)):
        for p in range(1, PERIODS_PER_DAY+1):
            for r in lecture_rooms:
                assign_th[(sid, subj, d_idx, p, r["id"])] = model.NewBoolVar(f"th_{sid}_{subj}_{d_idx}_{p}_{r['id']}")
        for p in range(1, PERIODS_PER_DAY - LAB_BLOCK + 2):
            for r in lab_rooms:
                assign_lab[(sid, subj, d_idx, p, r["id"])] = model.NewBoolVar(f"lab_{sid}_{subj}_{d_idx}_{p}_{r['id']}")

# required counts
for t in tasks:
    sid, subj = t["sectionId"], t["subjectId"]
    fid = task_faculty.get((sid, subj))
    if fid is None: continue
    th_vars = [v for k,v in assign_th.items() if k[0]==sid and k[1]==subj]
    model.Add(sum(th_vars) == t["theory_periods"])
    lab_vars = [v for k,v in assign_lab.items() if k[0]==sid and k[1]==subj]
    model.Add(sum(lab_vars) == t["lab_sessions"])

# room clash
slot_vars = defaultdict(list)
for (sid, subj, d_idx, p, r), v in assign_th.items():
    slot_vars[(d_idx,p,r)].append(v)
for (sid, subj, d_idx, p, r), v in assign_lab.items():
    slot_vars[(d_idx,p,r)].append(v)
    slot_vars[(d_idx,p+1,r)].append(v)
for vars_list in slot_vars.values():
    model.Add(sum(vars_list) <= 1)

# section clash
for sec in [s["id"] for s in sections_to_schedule]:
    for d_idx in range(len(DATES)):
        for p in range(1, PERIODS_PER_DAY+1):
            vars_here = []
            for (sid, subj, dd, pp, r), v in assign_th.items():
                if sid==sec and dd==d_idx and pp==p: vars_here.append(v)
            for (sid, subj, dd, pp, r), v in assign_lab.items():
                if sid==sec and dd==d_idx and (pp==p or pp+1==p): vars_here.append(v)
            if vars_here: model.Add(sum(vars_here) <= 1)

# faculty clash
fac_slot = defaultdict(list)
for (sid, subj, d_idx, p, r), v in assign_th.items():
    fid = task_faculty.get((sid, subj))
    if fid: fac_slot[(fid,d_idx,p)].append(v)
for (sid, subj, d_idx, p, r), v in assign_lab.items():
    fid = task_faculty.get((sid, subj))
    if fid:
        fac_slot[(fid,d_idx,p)].append(v)
        fac_slot[(fid,d_idx,p+1)].append(v)
for vars_list in fac_slot.values():
    model.Add(sum(vars_list) <= 1)

# faculty daily min/max
fac_day_vars = defaultdict(list)
for (fid, d_idx, p), vars_list in fac_slot.items():
    fac_day_vars[(fid,d_idx)].extend(vars_list)
for (fid,d_idx), vars_list in fac_day_vars.items():
    model.Add(sum(vars_list) <= MAX_FACULTY_PER_DAY)
    teach = model.NewBoolVar(f"teach_{fid}_{d_idx}")
    model.AddMaxEquality(teach, vars_list)
    model.Add(sum(vars_list) >= MIN_FACULTY_PER_DAY).OnlyEnforceIf(teach)

# faculty labs/week
lab_day_ind = {}
for (sid, subj, d_idx, p, r), v in assign_lab.items():
    fid = task_faculty.get((sid, subj))
    if not fid: continue
    wk = iso_week(DATES[d_idx])
    key = (fid,wk,d_idx)
    if key not in lab_day_ind:
        lab_day_ind[key] = model.NewBoolVar(f"labday_{fid}_{wk}_{d_idx}")
    model.Add(v <= lab_day_ind[key])
lab_inds_by_fwk = defaultdict(list)
for (fid,wk,d_idx), ind in lab_day_ind.items():
    lab_inds_by_fwk[(fid,wk)].append(ind)
for inds in lab_inds_by_fwk.values():
    model.Add(sum(inds) >= MIN_LABS_PER_FACULTY_PER_WEEK)

# section labs/week
sec_lab_day_ind = {}
for (sid, subj, d_idx, p, r), v in assign_lab.items():
    key = (sid, iso_week(DATES[d_idx]), d_idx)
    if key not in sec_lab_day_ind:
        sec_lab_day_ind[key] = model.NewBoolVar(f"seclab_{sid}_{key[1]}_{d_idx}")
    model.Add(v <= sec_lab_day_ind[key])
sec_week_inds = defaultdict(list)
for (sid,wk,d_idx), ind in sec_lab_day_ind.items():
    sec_week_inds[(sid,wk)].append(ind)
for inds in sec_week_inds.values():
    model.Add(sum(inds) >= MIN_LABS_PER_SECTION_PER_WEEK)

# section free period constraint
for sec in [s["id"] for s in sections_to_schedule]:
    for d_idx in range(len(DATES)):
        assigned = []
        for (sid, subj, dd, p, r), v in assign_th.items():
            if sid==sec and dd==d_idx: assigned.append(v)
        for (sid, subj, dd, p, r), v in assign_lab.items():
            if sid==sec and dd==d_idx: assigned.extend([v,v])
        if assigned:
            model.Add(sum(assigned) <= PERIODS_PER_DAY - FREE_PERIODS_PER_SECTION_PER_DAY)

# ---------- Objective: maximize continuity ----------
same_room_terms = []
for sec in [s["id"] for s in sections_to_schedule]:
    for d_idx in range(len(DATES)):
        for p in range(1, PERIODS_PER_DAY):
            for r in [rm["id"] for rm in rooms]:
                vars_p, vars_p1 = [], []
                for (sid, subj, dd, pp, rr), v in assign_th.items():
                    if sid==sec and dd==d_idx and rr==r:
                        if pp==p: vars_p.append(v)
                        if pp==p+1: vars_p1.append(v)
                for (sid, subj, dd, pp, rr), v in assign_lab.items():
                    if sid==sec and dd==d_idx and rr==r:
                        if pp==p: vars_p.extend([v,v])  # occupies p and p+1
                        if pp==p-1: vars_p1.append(v)
                if vars_p and vars_p1:
                    overlap = model.NewBoolVar(f"same_{sec}_{d_idx}_{p}_{r}")
                    model.Add(sum(vars_p) >= 1).OnlyEnforceIf(overlap)
                    model.Add(sum(vars_p1) >= 1).OnlyEnforceIf(overlap)
                    same_room_terms.append(overlap)
model.Maximize(sum(same_room_terms))

# ---------- Solve ----------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT_SECONDS
solver.parameters.num_search_workers = 8
print("Solving...")
res = solver.Solve(model)
print("Status:", solver.StatusName(res))

# ---------- Outputs ----------
section_timetable = defaultdict(lambda: defaultdict(lambda: {str(p): {"free": True} for p in range(1, PERIODS_PER_DAY+1)}))
faculty_alloc = defaultdict(lambda: {"allocations": [], "total_periods": 0, "lab_sessions": 0})

for (sid, subj, d_idx, p, r), v in assign_th.items():
    if solver.Value(v):
        section_timetable[sid][DATES[d_idx]][str(p)] = {
            "subjectId": subj,
            "facultyId": task_faculty.get((sid,subj)),
            "roomId": r,
            "type": "theory"
        }
        fid = task_faculty.get((sid,subj))
        faculty_alloc[fid]["allocations"].append({"date": DATES[d_idx], "period": p, "sectionId": sid, "subjectId": subj, "roomId": r, "type": "theory"})
        faculty_alloc[fid]["total_periods"] += 1

for (sid, subj, d_idx, p, r), v in assign_lab.items():
    if solver.Value(v):
        for pp in [p,p+1]:
            section_timetable[sid][DATES[d_idx]][str(pp)] = {
                "subjectId": subj,
                "facultyId": task_faculty.get((sid,subj)),
                "roomId": r,
                "type": "lab"
            }
        fid = task_faculty.get((sid,subj))
        faculty_alloc[fid]["allocations"].append({"date": DATES[d_idx], "period": p, "sectionId": sid, "subjectId": subj, "roomId": r, "type":"lab"})
        faculty_alloc[fid]["total_periods"] += LAB_BLOCK
        faculty_alloc[fid]["lab_sessions"] += 1

with open("section_timetable.json","w") as f:
    json.dump(section_timetable, f, indent=2)
with open("faculty_allocations.json","w") as f:
    json.dump(faculty_alloc, f, indent=2)

print("✅ Wrote section_timetable.json and faculty_allocations.json")
if unassigned:
    print("⚠️ Manual mapping required:", unassigned)
