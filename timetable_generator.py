#!/usr/bin/env python3
"""
Timetable Generator (CP-SAT) with Elastic Debug Mode.

- Modular constraint families so we can skip one family at a time to
  identify the infeasible constraint.
- Input files (in "input" directory):
    - aiml-faculty-detailed.json
    - semester-subjects.json
    - department-sections.json
    - classrooms.json
    - all_semesters_net_dates.json
- Outputs (in "output" directory):
    - section_timetable.json
    - faculty_allocations.json
    - timetable_model.proto (if infeasible/debug)
    - solver_stats.txt
"""
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from math import ceil
from ortools.sat.python import cp_model
from datetime import datetime, timedelta


# -----------------------
# Config
# -----------------------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
PERIODS_PER_DAY = 8
DAYS_PER_WEEK = 6
TIME_LIMIT_SECONDS = 60  # per solve attempt (elastic mode uses short attempts)
NUM_WORKERS = 4
PERIOD_MINUTES = 50
COVERAGE_TOLERANCE = (0.8, 1.2)  # 90% - 110% tolerance
# time config for per-period start time calculation
PERIOD_START_TIME = "08:50"   # first period start (HH:MM)


# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TimetableElastic")

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
    if hours is None:
        return 0
    minutes = hours * 60
    return int(ceil(minutes / PERIOD_MINUTES))

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Timetable class
# -----------------------
class TimetableFull:
    def __init__(self, input_dir=INPUT_DIR):
        ensure_dirs()
        self.input_dir = Path(input_dir)
        self.model = None
        self.solver = None
        self.reset_model()

        # raw data
        self.faculty = {}
        self.subjects = {}
        self.sections = {}
        self.classrooms = {}
        self.working_dates = []

        # maps
        self.subj_to_i = {}
        self.i_to_subj = {}
        self.fac_to_i = {}
        self.i_to_fac = {}
        self.room_to_i = {}
        self.i_to_room = {}

        # variables & caches
        self.faculty_assignment = {}  # (section, subj_int) -> IntVar
        self.slot_vars = defaultdict(lambda: defaultdict(dict))  # section -> d -> p -> (s_var,f_var,r_var,is_free,is_lab)
        self.slot_is_sub = {}  # (section,d,p,subj_int) -> BoolVar
        self.elective_groups = defaultdict(list)
        self.elective_active = {}  # (semester,group,d,p) -> BoolVar

    def reset_model(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
        self.solver.parameters.num_search_workers = NUM_WORKERS

    def load_inputs(self):
        logger.info("Loading inputs from %s", self.input_dir)
        if not self.input_dir.exists():
            logger.error("Input directory not found: %s", self.input_dir)
            return False
        try:
            with open(self.input_dir / "aiml-faculty-detailed.json") as f:
                self.faculty = {fobj['facultyId']: fobj for fobj in json.load(f)}
            with open(self.input_dir / "semester-subjects.json") as f:
                self.subjects = {s['subjectId']: s for s in json.load(f)}
            with open(self.input_dir / "department-sections.json") as f:
                self.sections = {sec['id']: sec for sec in json.load(f)}
            with open(self.input_dir / "classrooms.json") as f:
                self.classrooms = {r['id']: r for r in json.load(f)}
            with open(self.input_dir / "all_semesters_net_dates.json") as f:
                dates_data = json.load(f)
                if dates_data:
                    # choose first semester net dates by default (same as original script)
                    self.working_dates = dates_data[0].get('netDates', [])
        except Exception as e:
            logger.exception("Failed to load inputs: %s", e)
            return False

        # maps for ints
        self.subj_to_i, self.i_to_subj = to_int_map(self.subjects.values(), 'subjectId')
        self.fac_to_i, self.i_to_fac = to_int_map(self.faculty.values(), 'facultyId')
        self.room_to_i, self.i_to_room = to_int_map(self.classrooms.values(), 'id')

        logger.info("Loaded: subjects=%d faculty=%d sections=%d rooms=%d working_dates=%d",
                    len(self.subj_to_i), len(self.fac_to_i), len(self.sections), len(self.room_to_i), len(self.working_dates))
        return True

    # -----------------------
    # Constraint family builders
    # Each builder accepts a flag `enabled` (True/False).
    # -----------------------
    def build_elective_groups(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] build_elective_groups")
            return
        logger.info("Building elective groups...")
        self.elective_groups.clear()
        for subj_orig, subj in self.subjects.items():
            if subj.get('isElective') and subj.get('elective_group_name'):
                sem = subj.get('semester')
                group = subj.get('elective_group_name')
                subj_int = self.subj_to_i[subj_orig]
                self.elective_groups[(sem, group)].append(subj_int)
        logger.info("Found %d elective groups", len(self.elective_groups))

    def create_faculty_assignment_vars(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] create_faculty_assignment_vars")
            return
        logger.info("Creating faculty assignment variables per (section,subject)")
        subject_eligible_fac = {}
        # fuzzy mapping of faculty subjects -> subjectName
        for subj_orig, subj in self.subjects.items():
            subj_name = subj.get('subjectName', '').lower()
            elig = []
            for fac_orig, fac in self.faculty.items():
                fac_int = self.fac_to_i[fac_orig]
                fac_subjects = [s.lower() for s in fac.get('subjects', []) if isinstance(s, str)]
                can = any(ss in subj_name or subj_name in ss for ss in fac_subjects) or (subj_orig in fac.get('subjects', []))
                if can:
                    elig.append(fac_int)
            if not elig:
                # fallback to all faculty (keeps model feasible but reveals bad mapping)
                logger.warning("No direct faculty matched for subj %s (%s). Allowing all faculty as fallback.", subj_orig, subj.get('subjectName'))
                elig = list(self.i_to_fac.keys())
            subject_eligible_fac[self.subj_to_i[subj_orig]] = elig

        for section_id in self.sections:
            for subj_orig in self.subjects:
                subj_int = self.subj_to_i[subj_orig]
                domain = subject_eligible_fac.get(subj_int, list(self.i_to_fac.keys()))
                var = self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain),
                                                     f"assign_fac_sec{section_id}_sub{subj_int}")
                self.faculty_assignment[(section_id, subj_int)] = var
        logger.info("Created %d faculty assignment variables", len(self.faculty_assignment))

    def create_slot_vars(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] create_slot_vars")
            return
        logger.info("Creating slot variables (subject/fac/room) and free flags")
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

        logger.info("Created slot variables for approx %d slots", len(self.sections) * DAYS_PER_WEEK * PERIODS_PER_DAY)

    def link_subject_to_assigned_faculty(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] link_subject_to_assigned_faculty")
            return
        logger.info("Linking slot faculty to per-(section,subject) assignment and building slot_is_sub booleans")
        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    s_var, f_var, _, is_free, is_lab = self.slot_vars[section_id][d][p]
                    for subj_int in self.i_to_subj.keys():
                        key = (section_id, d, p, subj_int)
                        b = self.model.NewBoolVar(f"slot_is_sec{section_id}_d{d}_p{p}_sub{subj_int}")
                        self.model.Add(s_var == subj_int).OnlyEnforceIf(b)
                        self.model.Add(s_var != subj_int).OnlyEnforceIf(b.Not())
                        self.slot_is_sub[key] = b
                        # link faculty: if this slot has subj_int then slot f_var == faculty_assignment[(section,subj_int)]
                        assign_var = self.faculty_assignment[(section_id, subj_int)]
                        self.model.Add(f_var == assign_var).OnlyEnforceIf(b)

    def add_subject_room_allowed(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] add_subject_room_allowed")
            return
        logger.info("Adding allowed (subject,room) combos and linking is_lab booleans")
        lab_subj_ints = set()
        for subj_orig, subj in self.subjects.items():
            if subj.get('practicalHours', 0) > 0 or 'lab' in subj.get('subjectName', '').lower():
                lab_subj_ints.add(self.subj_to_i[subj_orig])
        all_subj_ints = set(self.i_to_subj.keys())

        lab_rooms = [i for i, r in self.i_to_room.items() if self.classrooms[r].get('type', '').lower() == 'lab']
        theory_rooms = [i for i, r in self.i_to_room.items() if self.classrooms[r].get('type', '').lower() == 'classroom']

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

        for section_id in self.sections:
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    s_var, _, r_var, is_free, is_lab = self.slot_vars[section_id][d][p]
                    self.model.Add(r_var == 0).OnlyEnforceIf(is_free)

                    # is_lab detection (OR of lab subject booleans)
                    if lab_subj_ints:
                        lab_reifs = []
                        for s_int in lab_subj_ints:
                            br = self.model.NewBoolVar(f"slot_{section_id}_{d}_{p}_is_sub_{s_int}")
                            self.model.Add(s_var == s_int).OnlyEnforceIf(br)
                            self.model.Add(s_var != s_int).OnlyEnforceIf(br.Not())
                            lab_reifs.append(br)
                        self.model.AddBoolOr(lab_reifs).OnlyEnforceIf(is_lab)
                        for br in lab_reifs:
                            self.model.Add(br == 0).OnlyEnforceIf(is_lab.Not())
                    else:
                        self.model.Add(is_lab == 0)

                    # allowed assignments (subject, room)
                    self.model.AddAllowedAssignments([s_var, r_var], allowed_list)

    def build_elective_active_vars(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] build_elective_active_vars")
            return
        logger.info("Building elective active vars for elective groups (if any)")
        self.elective_active.clear()
        for (sem, group), subj_list in self.elective_groups.items():
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    var = self.model.NewBoolVar(f"elective_active_sem{sem}_grp{group}_d{d}_p{p}")
                    self.elective_active[(sem, group, d, p)] = var

        # Enforce: if elective_active True then every section of that semester must have some subject from that group
        for (sem, group), subj_list in self.elective_groups.items():
            # Build subject bools per slot for sections belonging to this semester
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    active_var = self.elective_active[(sem, group, d, p)]
                    # For each section in this semester:
                    for section_id, sec in self.sections.items():
                        if sec.get('year') != sem:  # note: assumes semester stored in 'year' - adapt if needed
                            continue
                        # create bool: section_has_group_sub = OR(slot_is_sub for subj in subj_list)
                        group_reifs = []
                        for subj_int in subj_list:
                            b = self.slot_is_sub[(section_id, d, p, subj_int)]
                            group_reifs.append(b)
                        # section_has_group_sub = model.NewBoolVar...
                        if group_reifs:
                            self.model.AddBoolOr(group_reifs).OnlyEnforceIf(active_var)
                            # If elective active, no other non-group subject should be scheduled. (Enforce later via constraints)
        # NOTE: We keep elective semantics light here; main goal is to allow toggle for debugging.

    def add_no_double_booking(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] add_no_double_booking")
            return
        logger.info("Adding no-double-booking constraints: faculty/room/section")
        # For each timeslot, a faculty can't be in two places & room can't host two sections
        for d in range(DAYS_PER_WEEK):
            for p in range(PERIODS_PER_DAY):
                # Faculty: collect slot f_vars across sections; for each faculty id, at most one section can have that faculty at (d,p)
                # We achieve "all different" via pairwise inequality reified style
                fac_at_slot = {}
                room_at_slot = {}
                for section_id in self.sections:
                    _, f_var, r_var, is_free, is_lab = self.slot_vars[section_id][d][p]
                    # For each faculty integer, create bool equals (expensive). Instead use pairwise constraints:
                    for other_section in self.sections:
                        if other_section <= section_id:
                            continue
                        _, f_var2, r_var2, _, _ = self.slot_vars[other_section][d][p]
                        # if both non-free and faculties equal -> forbidden
                        b_both_nonfree = self.model.NewBoolVar(f"both_nonfree_{section_id}_{other_section}_{d}_{p}")
                        s1 = self.slot_vars[section_id][d][p][0]
                        s2 = self.slot_vars[other_section][d][p][0]
                        self.model.Add(s1 != 0).OnlyEnforceIf(b_both_nonfree)
                        self.model.Add(s2 != 0).OnlyEnforceIf(b_both_nonfree)
                        # if both_nonfree then f_var != f_var2
                        self.model.Add(f_var != f_var2).OnlyEnforceIf(b_both_nonfree)

                        # room conflicts: if both non-free then room_var != room_var2
                        self.model.Add(r_var != r_var2).OnlyEnforceIf(b_both_nonfree)

    def add_coverage_constraints(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] add_coverage_constraints")
            return
        logger.info("Adding coverage constraints (ensure subject hours mapped to slots approx within tolerance)")
        # Estimate total available periods for subject per semester: sections * days*periods * weeks approximated by working_dates
        total_working_days = len(self.working_dates) if self.working_dates else (DAYS_PER_WEEK * 16)  # fallback
        total_periods_available = total_working_days * PERIODS_PER_DAY

        # We'll enforce per-subject total assigned slots across all sections to be within tolerance of required periods
        # Compute required periods per subject: totalHours -> periods
        subj_required_periods = {}
        for subj_orig, subj in self.subjects.items():
            periods = hours_to_periods(subj.get('totalHours', 0))
            subj_required_periods[self.subj_to_i[subj_orig]] = periods

        # For each subject, count total slots assigned across all sections (weeks are not explicitly modeled; we map per-day/week pattern)
        # Because model is per-week pattern (DAYS_PER_WEEK * PERIODS_PER_DAY), we scale requirement to per-week by dividing by number of weeks.
        # For simplicity: require each subject to appear between tol_min and tol_max times per WEEK pattern where:
        # weeks_estimate = max(1, total_working_days // DAYS_PER_WEEK)
        weeks_estimate = max(1, total_working_days // DAYS_PER_WEEK)
        logger.info("Total working days=%d weeks_estimate=%d", total_working_days, weeks_estimate)

        for subj_int, periods in subj_required_periods.items():
            min_slots = int(ceil((periods * COVERAGE_TOLERANCE[0]) / weeks_estimate))
            max_slots = int(ceil((periods * COVERAGE_TOLERANCE[1]) / weeks_estimate))
            # count occurrences over (section,d,p)
            occurrence_bools = []
            for section_id in self.sections:
                for d in range(DAYS_PER_WEEK):
                    for p in range(PERIODS_PER_DAY):
                        b = self.slot_is_sub[(section_id, d, p, subj_int)]
                        occurrence_bools.append(b)
            if occurrence_bools:
                # Create int var = sum of bools
                occ_sum = sum(occurrence_bools)  # OR-Tools accepts sum of BoolVar
                # Enforce bounds (as linear constraints)
                self.model.Add(sum(occurrence_bools) >= min_slots)
                self.model.Add(sum(occurrence_bools) <= max_slots)
                logger.debug("Subject %s requires per-week slots in [%d, %d]", self.i_to_subj[subj_int], min_slots, max_slots)

    def add_faculty_load_constraints(self, enabled=True):
        if not enabled:
            logger.info("[SKIP] add_faculty_load_constraints")
            return
        logger.info("Adding faculty load constraints (no more than X slots/day and at least Y slots/day if needed)")
        # Example: a faculty can teach at most 4 slots/day, min 0. These are tunable; for debug we keep it moderate.
        MAX_SLOTS_PER_FACULTY_PER_DAY = 4
        for fac_int in self.i_to_fac:
            for d in range(DAYS_PER_WEEK):
                # collect bools across sections whether fac_int is assigned at (d)
                fac_presence = []
                for section_id in self.sections:
                    _, f_var, _, is_free, _ = self.slot_vars[section_id][d][0]  # we'll iterate p below; create per (section,p)
                # Build sum for all sections and periods
                presence_bools = []
                for section_id in self.sections:
                    for p in range(PERIODS_PER_DAY):
                        _, f_var, _, is_free, _ = self.slot_vars[section_id][d][p]
                        b = self.model.NewBoolVar(f"fac{fac_int}_sec{section_id}_d{d}_p{p}_is")
                        self.model.Add(f_var == fac_int).OnlyEnforceIf(b)
                        self.model.Add(f_var != fac_int).OnlyEnforceIf(b.Not())
                        presence_bools.append(b)
                if presence_bools:
                    self.model.Add(sum(presence_bools) <= MAX_SLOTS_PER_FACULTY_PER_DAY)

    # -----------------------
    # Solve & reporting
    # -----------------------
    def solve_and_report(self):
        logger.info("Solving model...")
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
        self.solver.parameters.num_search_workers = NUM_WORKERS
        status = self.solver.Solve(self.model)

        status_name = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
                       cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN"}.get(status, str(status))
        logger.info("Solver status: %s", status_name)
        # dump stats
        stats = self.solver.ResponseStats()
        with open(OUTPUT_DIR / "solver_stats.txt", "w") as sf:
            sf.write(f"Status: {status_name}\n")
            sf.write(stats + "\n")

        if status == cp_model.INFEASIBLE:
            # export model proto for external debugging
            try:
                proto = self.model.SerializeToString()
                with open(OUTPUT_DIR / "timetable_model.proto", "wb") as pf:
                    pf.write(proto)
                logger.error("Model infeasible. Serialized CP model to %s", OUTPUT_DIR / "timetable_model.proto")
            except Exception as e:
                logger.exception("Failed to write model proto: %s", e)
            return False

        # build outputs
        section_timetable = {}
        for section_id in self.sections:
            section_timetable[section_id] = []
            for d in range(DAYS_PER_WEEK):
                for p in range(PERIODS_PER_DAY):
                    s_var, f_var, r_var, is_free, is_lab = self.slot_vars[section_id][d][p]
                    subj_val = int(self.solver.Value(s_var))
                    fac_val = int(self.solver.Value(f_var))
                    room_val = int(self.solver.Value(r_var))
                    # --- begin replacement block ---
                    # raw ids
                    subj_val = int(self.solver.Value(s_var))
                    fac_val = int(self.solver.Value(f_var))
                    room_val = int(self.solver.Value(r_var))

                    # subject and faculty ids (string IDs used elsewhere)
                    subject_id = self.i_to_subj.get(subj_val) if subj_val != 0 else None
                    faculty_id = self.i_to_fac.get(fac_val) if fac_val != 0 else None
                    room_id = self.i_to_room.get(room_val) if room_val != 0 else None

                    # subject metadata
                    subject_name = None
                    is_elective = False
                    is_lab = False
                    if subject_id:
                        subj_obj = self.subjects.get(subject_id, {})
                        subject_name = subj_obj.get("subjectName") or subj_obj.get("name") or None
                        is_elective = bool(subj_obj.get("isElective", False))
                        # detect lab either via practicalHours or 'lab' in name
                        is_lab = (subj_obj.get("practicalHours", 0) > 0) or ("lab" in (subject_name or "").lower())

                    # faculty metadata (try common name keys safely)
                    faculty_name = None
                    if faculty_id:
                        fac_obj = self.faculty.get(faculty_id, {})
                        faculty_name = fac_obj.get("name") or fac_obj.get("fullName") or fac_obj.get("facultyName") or fac_obj.get("displayName")

                    # date calculation:
                    date_str = None
                    if self.working_dates:
                        # Prefer parsing first working date and adding day offset (robust if working_dates contains ISO dates)
                        try:
                            base = datetime.fromisoformat(self.working_dates[0])
                            date_dt = base + timedelta(days=d)
                            date_str = date_dt.date().isoformat()
                        except Exception:
                            # fallback: if working_dates has at least DAYS_PER_WEEK entries, use index by day
                            if len(self.working_dates) > d:
                                date_str = self.working_dates[d]
                            else:
                                date_str = None

                    # time calculation for the period (HH:MM)
                    time_str = None
                    try:
                        start_dt = datetime.strptime(PERIOD_START_TIME, "%H:%M")
                        slot_dt = start_dt + timedelta(minutes=PERIOD_MINUTES * p)
                        time_str = slot_dt.strftime("%H:%M")
                    except Exception:
                        time_str = None

                    entry = {
                        "day": d,
                        "date": date_str,
                        "time": time_str,
                        "period": p,
                        "subject": subject_id,
                        "subjectName": subject_name,
                        "isElective": is_elective,
                        "isLab": is_lab,
                        "faculty": faculty_id,
                        "facultyName": faculty_name,
                        "room": room_id,
                        "is_free": bool(self.solver.Value(is_free))
                    }
                    # --- end replacement block ---

                    section_timetable[section_id].append(entry)

        # faculty allocations: list assignments (section,subject->faculty)
        faculty_allocations = []
        for (section_id, subj_int), var in self.faculty_assignment.items():
            try:
                fac_val = int(self.solver.Value(var))
            except Exception:
                fac_val = None
            faculty_allocations.append({
                "section": section_id,
                "subject": self.i_to_subj.get(subj_int),
                "faculty": self.i_to_fac.get(fac_val) if fac_val else None
            })

        with open(OUTPUT_DIR / "section_timetable.json", "w") as f:
            json.dump(section_timetable, f, indent=2)
        with open(OUTPUT_DIR / "faculty_allocations.json", "w") as f:
            json.dump(faculty_allocations, f, indent=2)

        logger.info("Wrote outputs to %s", OUTPUT_DIR)
        return True

    # -----------------------
    # Elastic debug runner
    # -----------------------
    def run_elastic_debug(self):
        """
        Attempt solving with all constraints and then re-run skipping one family at a time.
        Returns dict: family -> (status_name, made_feasible_bool)
        """
        logger.info("=== Running Elastic Debug ===")
        family_builders = {
            "election_groups": lambda enabled: self.build_elective_groups(enabled=enabled),
            "faculty_assignment": lambda enabled: self.create_faculty_assignment_vars(enabled=enabled),
            "slot_vars": lambda enabled: self.create_slot_vars(enabled=enabled),
            "link_faculty": lambda enabled: self.link_subject_to_assigned_faculty(enabled=enabled),
            "room_lab": lambda enabled: self.add_subject_room_allowed(enabled=enabled),
            "elective_active": lambda enabled: self.build_elective_active_vars(enabled=enabled),
            "no_double_booking": lambda enabled: self.add_no_double_booking(enabled=enabled),
            "coverage": lambda enabled: self.add_coverage_constraints(enabled=enabled),
            "faculty_load": lambda enabled: self.add_faculty_load_constraints(enabled=enabled),
        }

        results = {}

        # baseline run (all enabled)
        logger.info("--- Baseline run (all constraints enabled) ---")
        self.reset_model()
        # re-create containers
        self.faculty_assignment.clear()
        self.slot_vars.clear()
        self.slot_is_sub.clear()
        self.elective_active.clear()
        # call all builders with enabled=True
        for name, builder in family_builders.items():
            try:
                builder(True)
            except Exception as e:
                logger.exception("Error running builder %s: %s", name, e)
        baseline_ok = self.solve_and_report()
        results["baseline"] = {"feasible": baseline_ok}
        if baseline_ok:
            logger.info("Baseline is feasible. No need for debug skipping.")
            return results

        # Now try skipping each family in turn
        for skip_name in family_builders.keys():
            logger.info("=== Debug attempt: SKIP %s ===", skip_name)
            self.reset_model()
            # clear vars caches
            self.faculty_assignment.clear()
            self.slot_vars.clear()
            self.slot_is_sub.clear()
            self.elective_active.clear()

            # Important: ensure that basic structures exist: faculty_assignment & slot_vars are needed for others
            # We'll call faculty_assignment and slot_vars builders as normal unless they are the ones being skipped.
            for name, builder in family_builders.items():
                enabled = (name != skip_name)
                try:
                    builder(enabled)
                except Exception as e:
                    logger.exception("Error building %s (enabled=%s): %s", name, enabled, e)

            ok = self.solve_and_report()
            results[skip_name] = {"feasible": ok}
            if ok:
                logger.warning("✅ Skipping '%s' made model feasible.", skip_name)
            else:
                logger.info("❌ Skipping '%s' did not fix infeasibility.", skip_name)

        logger.info("Elastic debug complete. Results: %s", results)
        return results

# -----------------------
# Main
# -----------------------
def main():
    timetable = TimetableFull(INPUT_DIR)
    if not timetable.load_inputs():
        logger.error("Failed to load inputs; exiting.")
        sys.exit(1)

    # Mode selection
    DEBUG_ELASTIC = True  # set True to run elastic debug (skips one family at a time)
    if DEBUG_ELASTIC:
        results = timetable.run_elastic_debug()
        # write results summary
        with open(OUTPUT_DIR / "elastic_debug_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Elastic debug summary written to %s", OUTPUT_DIR / "elastic_debug_summary.json")
        return

    # Normal run (all constraints)
    timetable.reset_model()
    # create constraints
    timetable.build_elective_groups(enabled=True)
    timetable.create_faculty_assignment_vars(enabled=True)
    timetable.create_slot_vars(enabled=True)
    timetable.link_subject_to_assigned_faculty(enabled=True)
    timetable.add_subject_room_allowed(enabled=True)
    timetable.build_elective_active_vars(enabled=True)
    timetable.add_no_double_booking(enabled=True)
    timetable.add_coverage_constraints(enabled=True)
    timetable.add_faculty_load_constraints(enabled=True)

    ok = timetable.solve_and_report()
    if not ok:
        logger.error("Final run infeasible. Inspect %s/timetable_model.proto and solver_stats.txt", OUTPUT_DIR)
        sys.exit(2)
    logger.info("Timetable generated successfully.")

if __name__ == "__main__":
    main()
