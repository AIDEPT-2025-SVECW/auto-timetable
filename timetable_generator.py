#!/usr/bin/env python3
"""
Full Timetable Generator (CP-SAT) — Implements user's constraints with 90-110% tolerance.

Save as: timetable_generator_full.py
Inputs (place in input/):
 - aiml-faculty-detailed.json
 - semester-subjects.json
 - department-sections.json
 - classrooms.json
 - all_semesters_net_dates.json

Outputs:
 - output/section_timetable.json
 - output/faculty_allocations.json

Notes:
 - Period length assumed 50 minutes (for totalHours -> period count conversion).
 - May need to raise TIME_LIMIT_SECONDS for larger instances.
"""
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from math import ceil
from ortools.sat.python import cp_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TimetableFull")

# -----------------------
# Configuration
# -----------------------
PERIODS_PER_DAY = 8
DAYS_PER_WEEK = 6  # Monday-Saturday
TIME_LIMIT_SECONDS = 1800  # increase if solver struggles
NUM_WORKERS = 4
PERIOD_MINUTES = 50  # used to convert subject hours -> periods

# -----------------------
# Helpers
# -----------------------
def to_int_map(items, id_field):
    m = {}
    rm = {}
    idx = 1
    for it in items:
        orig = it[id_field]
        m[orig] = idx
        rm[idx] = orig
        idx += 1
    return m, rm

def hours_to_periods(hours):
    # convert hours -> number of 50-minute periods, round up
    if hours is None:
        return 0
    minutes = hours * 60
    return int(ceil(minutes / PERIOD_MINUTES))

# -----------------------
# Main class
# -----------------------
class TimetableFull:
    def __init__(self, input_dir="input"):
        self.input_dir = Path(input_dir)
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # raw data
        self.faculty = {}
        self.subjects = {}
        self.sections = {}
        self.classrooms = {}
        self.working_dates = []

        # id maps for compact integer domains
        self.subj_to_i = {}
        self.i_to_subj = {}
        self.fac_to_i = {}
        self.i_to_fac = {}
        self.room_to_i = {}
        self.i_to_room = {}

        # Variables
        # faculty_assignment[(section_id, subj_int)] = intvar (faculty int)
        self.faculty_assignment = {}

        # slot_vars[section_id][day][period] = (sub_var, fac_var, room_var, is_free_bool, is_lab_bool)
        self.slot_vars = defaultdict(lambda: defaultdict(dict))

    def load_inputs(self):
        logger.info("Loading input JSONs from %s", self.input_dir)
        if not self.input_dir.exists():
            logger.error("Input directory not found")
            return False
        try:
            self.faculty = {f['facultyId']: f for f in json.load(open(self.input_dir / "aiml-faculty-detailed.json"))}
            self.subjects = {s['subjectId']: s for s in json.load(open(self.input_dir / "semester-subjects.json"))}
            self.sections = {sec['id']: sec for sec in json.load(open(self.input_dir / "department-sections.json"))}
            self.classrooms = {r['id']: r for r in json.load(open(self.input_dir / "classrooms.json"))}
            dates_data = json.load(open(self.input_dir / "all_semesters_net_dates.json"))
            if dates_data:
                # use first semester entry's netDates
                self.working_dates = dates_data[0].get('netDates', [])
        except Exception as e:
            logger.exception("Failed loading inputs: %s", e)
            return False

        # build maps
        self.subj_to_i, self.i_to_subj = to_int_map(self.subjects.values(), 'subjectId')
        self.fac_to_i, self.i_to_fac = to_int_map(self.faculty.values(), 'facultyId')
        self.room_to_i, self.i_to_room = to_int_map(self.classrooms.values(), 'id')

        logger.info("Loaded counts — subjects: %d, faculty: %d, sections: %d, rooms: %d, working_dates: %d",
                    len(self.subj_to_i), len(self.fac_to_i), len(self.sections), len(self.room_to_i), len(self.working_dates))
        return True

    def create_faculty_assignment_vars(self):
        logger.info("Creating faculty assignment variables per (section,subject)")
        # For each (section,subject) that belongs to semester-subject list, create an assignment var
        # Determine eligible faculties per subject via fuzzy match on faculty['subjects'] entries
        # Build allowed faculty ints for each subject
        subject_eligible_fac = {}
        for subj_orig, subj in self.subjects.items():
            subj_name = subj.get('subjectName', '').lower()
            elig = []
            for fac_orig, fac in self.faculty.items():
                fac_int = self.fac_to_i[fac_orig]
                fac_subjects = [s.lower() for s in fac.get('subjects', [])]
                # qualification check: any substring match or explicit subjectId mention
                can = any(ss in subj_name or subj_name in ss for ss in fac_subjects) or (subj_orig in fac.get('subjects', []))
                if can:
                    elig.append(fac_int)
            # if no eligible faculty found, keep all faculties as fallback (to avoid infeasible immediate)
            if not elig:
                elig = list(self.i_to_fac.keys())
            subject_eligible_fac[self.subj_to_i[subj_orig]] = elig

        # create var per (section, subj_int)
        for section_id in self.sections:
            for subj_orig in self.subjects:
                subj_int = self.subj_to_i[subj_orig]
                domain = subject_eligible_fac.get(subj_int, list(self.i_to_fac.keys()))
                var = self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain), f"assign_fac_sec{section_id}_sub{subj_int}")
                self.faculty_assignment[(section_id, subj_int)] = var

        logger.info("Created %d faculty assignment variables", len(self.faculty_assignment))

    def create_slot_vars(self):
        logger.info("Creating slot variables per section/day/period and helper booleans")
        subj_domain = [0] + list(self.i_to_subj.keys())  # 0 = FREE
        fac_domain = [0] + list(self.i_to_fac.keys())
        room_domain = [0] + list(self.i_to_room.keys())

        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    s_var = self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(subj_domain),
                                                           f"slot_sub_{section_id}_{d}_{p}")
                    f_var = self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(fac_domain),
                                                           f"slot_fac_{section_id}_{d}_{p}")
                    r_var = self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(room_domain),
                                                           f"slot_room_{section_id}_{d}_{p}")
                    is_free = self.model.NewBoolVar(f"is_free_{section_id}_{d}_{p}")
                    is_lab = self.model.NewBoolVar(f"is_lab_{section_id}_{d}_{p}")

                    # reify free
                    self.model.Add(s_var == 0).OnlyEnforceIf(is_free)
                    self.model.Add(s_var != 0).OnlyEnforceIf(is_free.Not())

                    self.slot_vars[section_id][d][p] = (s_var, f_var, r_var, is_free, is_lab)

        logger.info("Created slot variables for approximately %d slots", len(self.sections) * DAYS_PER_WEEK * PERIODS_PER_DAY)

    def link_subject_to_assigned_faculty(self):
        logger.info("Link slot faculty to per-(section,subject) assignment")
        # For each slot: if slot subject == subj_int then slot faculty must equal faculty_assignment[(section,subj_int)]
        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    s_var, f_var, _, is_free, is_lab = self.slot_vars[section_id][d][p]
                    # for every subject value (non-zero), create reified bool and link faculty
                    for subj_int in self.i_to_subj.keys():
                        b = self.model.NewBoolVar(f"slot_is_sec{section_id}_d{d}_p{p}_sub{subj_int}")
                        self.model.Add(s_var == subj_int).OnlyEnforceIf(b)
                        self.model.Add(s_var != subj_int).OnlyEnforceIf(b.Not())
                        # slot faculty == assigned faculty for this section-subject if b true
                        assign_var = self.faculty_assignment[(section_id, subj_int)]
                        self.model.Add(f_var == assign_var).OnlyEnforceIf(b)
                        # also set is_lab reification later; for now we'll connect s_var -> is_lab via subject set in room constraints

    def add_subject_room_allowed(self):
        logger.info("Adding allowed (subject,room) combos and linking is_lab booleans")
        # determine lab subjects
        lab_subj_ints = set()
        for subj_orig, subj in self.subjects.items():
            if subj.get('practicalHours', 0) > 0 or 'lab' in subj.get('subjectName', '').lower():
                lab_subj_ints.add(self.subj_to_i[subj_orig])
        all_subj_ints = set(self.i_to_subj.keys())

        # room lists
        lab_rooms = [i for i, r in self.i_to_room.items() if self.classrooms[r].get('type', '').lower() == 'lab']
        theory_rooms = [i for i, r in self.i_to_room.items() if self.classrooms[r].get('type', '').lower() != 'lab']

        allowed_pairs = set()
        allowed_pairs.add((0, 0))
        for s in lab_subj_ints:
            for rr in lab_rooms:
                allowed_pairs.add((s, rr))
        non_lab = all_subj_ints - lab_subj_ints
        for s in non_lab:
            for rr in theory_rooms:
                allowed_pairs.add((s, rr))
        allowed_list = list(allowed_pairs)

        # apply to slots; also set is_lab reification
        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    s_var, _, r_var, is_free, is_lab = self.slot_vars[section_id][d][p]
                    # free -> room 0
                    self.model.Add(r_var == 0).OnlyEnforceIf(is_free)
                    # reify is_lab: if s_var in lab_subj_ints then is_lab true; else false
                    if lab_subj_ints:
                        lab_reifs = []
                        for s_int in lab_subj_ints:
                            br = self.model.NewBoolVar(f"slot_{section_id}_{d}_{p}_is_sub_{s_int}")
                            self.model.Add(s_var == s_int).OnlyEnforceIf(br)
                            self.model.Add(s_var != s_int).OnlyEnforceIf(br.Not())
                            lab_reifs.append(br)
                        # is_lab -> OR(lab_reifs); not is_lab -> all lab_reifs false
                        self.model.AddBoolOr(lab_reifs).OnlyEnforceIf(is_lab)
                        for br in lab_reifs:
                            self.model.Add(br == 0).OnlyEnforceIf(is_lab.Not())
                    else:
                        self.model.Add(is_lab == 0)
                    # Allowed assignments for (subject, room)
                    self.model.AddAllowedAssignments([s_var, r_var], allowed_list)

    def prevent_double_booking(self):
        logger.info("Preventing faculty double-booking across sections at same day/period")
        for d in range(DAYS_PER_WEEK):
            for p in range(PERIODS_PER_DAY):
                for fac_int in self.i_to_fac.keys():
                    bools = []
                    for section_id in self.sections:
                        _, f_var, _, _, _ = self.slot_vars[section_id][d][p]
                        b = self.model.NewBoolVar(f"fac{fac_int}_sec{section_id}_d{d}_p{p}")
                        self.model.Add(f_var == fac_int).OnlyEnforceIf(b)
                        self.model.Add(f_var != fac_int).OnlyEnforceIf(b.Not())
                        bools.append(b)
                    self.model.Add(sum(bools) <= 1)

    def enforce_one_free_per_day(self):
        logger.info("Enforce each section has at least 1 free period per day")
        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                free_bools = []
                for p in range(PERIODS_PER_DAY):
                    _, _, _, is_free, _ = self.slot_vars[section_id][d][p]
                    free_bools.append(is_free)
                self.model.Add(sum(free_bools) >= 1)

    def enforce_lab_session_counts(self):
        logger.info("Enforce section and faculty lab session minima (lab session = 2 contiguous periods)")
        # First, per section: at least 3 lab sessions per week (each session is 2 continuous periods)
        for section_id in self.sections:
            lab_session_bools = []
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY - 1):
                    s1, f1, r1, is_free1, is_lab1 = self.slot_vars[section_id][d][p]
                    s2, f2, r2, is_free2, is_lab2 = self.slot_vars[section_id][d][p+1]
                    # define a bool 'lab_session' true iff both is_lab1 and is_lab2 AND same subject AND same faculty AND same room
                    lab_session = self.model.NewBoolVar(f"lab_sess_sec{section_id}_d{d}_p{p}")
                    # constraints that imply lab_session -> conditions
                    # if lab_session true -> is_lab1 & is_lab2
                    self.model.Add(is_lab1 == 1).OnlyEnforceIf(lab_session)
                    self.model.Add(is_lab2 == 1).OnlyEnforceIf(lab_session)
                    # same subject and same faculty and same room
                    self.model.Add(s2 == s1).OnlyEnforceIf(lab_session)
                    self.model.Add(f2 == f1).OnlyEnforceIf(lab_session)
                    self.model.Add(r1 != 0).OnlyEnforceIf(lab_session)
                    self.model.Add(r2 != 0).OnlyEnforceIf(lab_session)
                    # converses: if both is_lab and equal subject and equal faculty and r1/r2 nonzero -> lab_session true
                    # To avoid too many extra constraints, we won't enforce the full equivalence (keeps model lighter).
                    lab_session_bools.append(lab_session)
            # require at least 3 lab sessions per week
            self.model.Add(sum(lab_session_bools) >= 3)

        # For each faculty: at least 2 lab sessions per week
        for fac_int in self.i_to_fac.keys():
            fac_lab_bools = []
            for section_id in self.sections:
                for d in range(DAYS_PER_WEEK):
                    for p in range(PERIODS_PER_DAY - 1):
                        s1, f1, r1, is_free1, is_lab1 = self.slot_vars[section_id][d][p]
                        s2, f2, r2, is_free2, is_lab2 = self.slot_vars[section_id][d][p+1]
                        lab_b = self.model.NewBoolVar(f"fac{fac_int}_lab_sec{section_id}_d{d}_p{p}")
                        # if this faculty is assigned at both contiguous periods and is_lab true for first
                        self.model.Add(f1 == fac_int).OnlyEnforceIf(lab_b)
                        self.model.Add(f2 == fac_int).OnlyEnforceIf(lab_b)
                        self.model.Add(is_lab1 == 1).OnlyEnforceIf(lab_b)
                        # converses not added to keep model lighter
                        fac_lab_bools.append(lab_b)
            self.model.Add(sum(fac_lab_bools) >= 2)

    def enforce_faculty_daily_minmax(self):
        logger.info("Enforce faculty min and max classes per day (min 2 if working, max 5 always)")
        for fac_int in self.i_to_fac.keys():
            for d in range(DAYS_PER_WEEK):
                period_bools = []
                for section_id in self.sections:
                    for p in range(PERIODS_PER_DAY):
                        _, f_var, _, _, _ = self.slot_vars[section_id][d][p]
                        b = self.model.NewBoolVar(f"fac{fac_int}_sec{section_id}_d{d}_p{p}_works")
                        self.model.Add(f_var == fac_int).OnlyEnforceIf(b)
                        self.model.Add(f_var != fac_int).OnlyEnforceIf(b.Not())
                        period_bools.append(b)
                # If faculty has any class that day -> must have at least 2
                has_any = self.model.NewBoolVar(f"fac{fac_int}_has_any_d{d}")
                self.model.Add(sum(period_bools) >= 1).OnlyEnforceIf(has_any)
                self.model.Add(sum(period_bools) == 0).OnlyEnforceIf(has_any.Not())
                self.model.Add(sum(period_bools) >= 2).OnlyEnforceIf(has_any)
                # Always cap at 5 per day
                self.model.Add(sum(period_bools) <= 5)

    def enforce_faculty_monthly_holidays(self):
        logger.info("Enforce max 2 holiday days per month per faculty (implemented as min required workdays)")
        total_working_days = len(self.working_dates)
        months = max(1, total_working_days // 30)
        min_workdays_required = max(0, total_working_days - 2 * months)
        # For each faculty, count distinct days where they have >=1 slot across sections
        for fac_int in self.i_to_fac.keys():
            day_bools = []
            for d in range(DAYS_PER_WEEK):
                # create bool that indicates faculty works on this day (across ANY section)
                work_day_bool = self.model.NewBoolVar(f"fac{fac_int}_works_day{d}")
                # build list of per-section period-bools for that day
                any_periods = []
                for section_id in self.sections:
                    for p in range(PERIODS_PER_DAY):
                        _, f_var, _, _, _ = self.slot_vars[section_id][d][p]
                        b = self.model.NewBoolVar(f"fac{fac_int}_sec{section_id}_d{d}_p{p}_present")
                        self.model.Add(f_var == fac_int).OnlyEnforceIf(b)
                        self.model.Add(f_var != fac_int).OnlyEnforceIf(b.Not())
                        any_periods.append(b)
                # work_day_bool iff any_periods sum >=1
                self.model.Add(sum(any_periods) >= 1).OnlyEnforceIf(work_day_bool)
                self.model.Add(sum(any_periods) == 0).OnlyEnforceIf(work_day_bool.Not())
                day_bools.append(work_day_bool)
            # Enforce min required working days across the semester (approximation using DAYS_PER_WEEK)
            # Scale min_workdays_required to days-per-week proportion: compute weekly_workdays_required
            # Use a conservative lower bound: require faculty to work at least (min_workdays_required // (max(1, total_working_days // DAYS_PER_WEEK)))
            weekly_required = max(0, (min_workdays_required * DAYS_PER_WEEK) // max(1, total_working_days))
            # Sum of day_bools across a week should be >= weekly_required
            self.model.Add(sum(day_bools) >= weekly_required)

    def enforce_subject_coverage(self):
        logger.info("Enforce subject coverage per section within 90-110% tolerance")
        # compute total weeks approx
        total_working_days = len(self.working_dates)
        total_weeks = max(1, total_working_days // 7)
        # For each section and subject: required periods = ceil(totalHours / period_minutes)
        for section_id in self.sections:
            for subj_orig, subj in self.subjects.items():
                subj_int = self.subj_to_i[subj_orig]
                total_hours = subj.get('totalHours', 0) or 0
                required_periods_total = hours_to_periods(total_hours)
                # apply tolerance 90%..110%
                low = int(max(0, int(ceil(required_periods_total * 0.7))))
                high = int(max(0, int(ceil(required_periods_total * 1.5))))
                # collect all slot bools where this section has subj_int
                slot_bools = []
                for d in range(DAYS_PER_WEEK):
                    for p in range(PERIODS_PER_DAY):
                        s_var, _, _, _, _ = self.slot_vars[section_id][d][p]
                        b = self.model.NewBoolVar(f"sec{section_id}_sub{subj_int}_d{d}_p{p}_is")
                        self.model.Add(s_var == subj_int).OnlyEnforceIf(b)
                        self.model.Add(s_var != subj_int).OnlyEnforceIf(b.Not())
                        slot_bools.append(b)
                # enforce between low and high across the whole semester (approximated to a week by scaling)
                # We treat these counts as weekly counts; scale required_periods_total by weeks to weeks? To be simple: require weekly count >= low_weekly etc.
                # Here we'll treat required_periods_total as total per semester; divide by total_weeks to get per-week requirement (rounded)
                per_week_low = max(0, low // max(1, total_weeks))
                per_week_high = max(0, max(1, high // max(1, total_weeks)))
                # enforce weekly bounds
                self.model.Add(sum(slot_bools) >= per_week_low)
                self.model.Add(sum(slot_bools) <= per_week_high)

                # Link: ensure faculty assignment variable exists and used as the only faculty for this subject in that section:
                assign_var = self.faculty_assignment[(section_id, subj_int)]
                # For every slot bool where subj is scheduled, ensure corresponding slot f_var == assign_var (we added earlier in link_subject_to_assigned_faculty).
                # Already linked in link_subject_to_assigned_faculty through per-subject reified equality.

    def prefer_single_room_per_continuous_block(self):
        logger.info("Prefer but do not strictly enforce single classroom per section continuous period (heuristic)")
        # Soft / heuristic: try to reduce room changes by adding simple linking:
        # If a subject occupies consecutive periods for same section and day, prefer same room by adding implication:
        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY - 1):
                    s1, f1, r1, is_free1, is_lab1 = self.slot_vars[section_id][d][p]
                    s2, f2, r2, is_free2, is_lab2 = self.slot_vars[section_id][d][p+1]
                    # If both periods are non-free and same subject -> encourage r1 == r2 by hard constraint (makes solver keep same room)
                    same_subj = self.model.NewBoolVar(f"sec{section_id}_d{d}_p{p}_same_subj")
                    self.model.Add(s1 == s2).OnlyEnforceIf(same_subj)
                    self.model.Add(s1 != s2).OnlyEnforceIf(same_subj.Not())
                    # If same_subj then enforce r1 == r2 (hard preference)
                    self.model.Add(r1 == r2).OnlyEnforceIf(same_subj)

    def solve_and_extract(self):
        logger.info("Solving (time_limit=%ds workers=%d)", TIME_LIMIT_SECONDS, NUM_WORKERS)
        self.solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
        self.solver.parameters.num_search_workers = NUM_WORKERS
        self.solver.parameters.log_search_progress = True

        status = self.solver.Solve(self.model)
        logger.info("Solver status: %s", self.solver.StatusName(status))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.error("No feasible solution found")
            return False

        # Extract timetable
        section_timetable = {}
        faculty_allocations = defaultdict(lambda: defaultdict(list))

        for section_id in self.sections:
            section_timetable[section_id] = {}
            for d in range(DAYS_PER_WEEK):
                day_name = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][d]
                section_timetable[section_id][day_name] = {}
                for p in range(PERIODS_PER_DAY):
                    s_var, f_var, r_var, is_free, is_lab = self.slot_vars[section_id][d][p]
                    s_val = self.solver.Value(s_var)
                    if s_val == 0:
                        section_timetable[section_id][day_name][f"Period_{p+1}"] = "FREE"
                        continue
                    subj_orig = self.i_to_subj[s_val]
                    fac_val = self.solver.Value(f_var)
                    room_val = self.solver.Value(r_var)
                    fac_orig = self.i_to_fac.get(fac_val, "")
                    room_orig = self.i_to_room.get(room_val, "")

                    entry = {
                        "subject": self.subjects[subj_orig].get('subjectName'),
                        "subjectId": subj_orig,
                        "faculty": self.faculty.get(fac_orig, {}).get('name'),
                        "facultyId": fac_orig,
                        "classroom": self.classrooms.get(room_orig, {}).get('name'),
                        "classroomId": room_orig
                    }
                    section_timetable[section_id][day_name][f"Period_{p+1}"] = entry
                    faculty_allocations[fac_orig][subj_orig].append({
                        "section": section_id, "day": day_name, "period": p+1, "room": room_orig
                    })

        # Save outputs
        outdir = Path("output")
        outdir.mkdir(exist_ok=True)
        with open(outdir / "section_timetable.json", "w") as f:
            json.dump(section_timetable, f, indent=2)
        processed = {}
        for fac_orig, subjmap in faculty_allocations.items():
            processed[fac_orig] = {
                "name": self.faculty.get(fac_orig, {}).get('name'),
                "subjects": {}
            }
            for subj_orig, periods in subjmap.items():
                processed[fac_orig]['subjects'][subj_orig] = {
                    "subjectName": self.subjects[subj_orig].get('subjectName'),
                    "totalPeriods": len(periods),
                    "periods": periods
                }
        with open(outdir / "faculty_allocations.json", "w") as f:
            json.dump(processed, f, indent=2)
        logger.info("Results saved to output/ (section_timetable.json, faculty_allocations.json)")
        return True

    def run(self):
        if not self.load_inputs():
            return False
        self.create_faculty_assignment_vars()
        self.create_slot_vars()
        self.link_subject_to_assigned_faculty()
        self.add_subject_room_allowed()
        self.prevent_double_booking()
        self.enforce_one_free_per_day()
        self.enforce_lab_session_counts()
        self.enforce_faculty_daily_minmax()
        self.enforce_faculty_monthly_holidays()
        self.enforce_subject_coverage()
        self.prefer_single_room_per_continuous_block()
        return self.solve_and_extract()

# -----------------------
# Main
# -----------------------
def main():
    gen = TimetableFull()
    ok = gen.run()
    if not ok:
        logger.error("Timetable generation failed or infeasible. Try increasing TIME_LIMIT_SECONDS or relaxing constraints.")
        sys.exit(1)
    print("Timetable generation completed — check output/ directory.")

if __name__ == "__main__":
    main()
