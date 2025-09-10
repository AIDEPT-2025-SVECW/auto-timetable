#!/usr/bin/env python3
"""
Automated Timetable Generation System
Uses Google OR-Tools to solve the complex constraint satisfaction problem
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import calendar

from ortools.sat.python import cp_model


@dataclass
class Faculty:
    id: str
    name: str
    department: str
    subjects: List[str]
    specialization: List[str]
    max_hours_per_day: int = 5
    min_hours_per_day: int = 2
    min_labs_per_week: int = 2
    max_holidays_per_month: int = 2


@dataclass
class Subject:
    code: str
    name: str
    semester: int
    hours_per_week: int
    is_lab: bool = False
    requires_continuous_hours: bool = False


@dataclass
class Section:
    id: str
    name: str
    year: int
    section: str
    total_students: int
    class_teacher: str


@dataclass
class Classroom:
    id: str
    name: str
    number: str
    type: str  # classroom, lab, conference, etc.
    capacity: int
    department: str


@dataclass
class TimeSlot:
    day: str
    period: int
    start_time: str
    end_time: str


class TimetableGenerator:
    def __init__(self, input_folder: str = "input"):
        self.input_folder = input_folder
        self.faculties = {}
        self.subjects = {}
        self.sections = {}
        self.classrooms = {}
        self.working_days = []
        self.time_slots = []
        self.model = cp_model.CpModel()
        self.variables = {}
        
        # Initialize time slots (8 periods per day)
        self._initialize_time_slots()
        self._load_data()
        
    def _initialize_time_slots(self):
        """Initialize daily time slots with breaks"""
        periods = [
            (1, "08:00", "08:50"),
            (2, "08:50", "09:40"),
            (3, "09:40", "10:30"),
            # Break 10:30-10:50
            (4, "10:50", "11:40"),
            (5, "11:40", "12:30"),
            # Lunch Break 12:30-13:50
            (6, "13:50", "14:40"),
            (7, "14:40", "15:30"),
            (8, "15:30", "16:20")
        ]
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        for day in days:
            for period, start, end in periods:
                self.time_slots.append(TimeSlot(day, period, start, end))
    
    def _load_data(self):
        """Load all input data from JSON files"""
        try:
            # Load faculty data
            with open(os.path.join(self.input_folder, "aiml-faculty-detailed.json"), 'r') as f:
                faculty_data = json.load(f)
                for faculty in faculty_data:
                    self.faculties[faculty['facultyId']] = Faculty(
                        id=faculty['facultyId'],
                        name=faculty['name'],
                        department=faculty['department'],
                        subjects=faculty['subjects'],
                        specialization=faculty['specialization']
                    )
            
            # Load subject data
            with open(os.path.join(self.input_folder, "semester-subjects.json"), 'r') as f:
                subjects_data = json.load(f)
                for subject in subjects_data:
                    # Estimate hours per week based on subject type
                    hours_per_week = self._estimate_subject_hours(subject['subjectName'])
                    is_lab = self._is_lab_subject(subject['subjectName'])
                    
                    self.subjects[subject['subjectCode']] = Subject(
                        code=subject['subjectCode'],
                        name=subject['subjectName'],
                        semester=subject['semester'],
                        hours_per_week=hours_per_week,
                        is_lab=is_lab,
                        requires_continuous_hours=is_lab
                    )
            
            # Load section data
            with open(os.path.join(self.input_folder, "department-sections.json"), 'r') as f:
                sections_data = json.load(f)
                for section in sections_data:
                    self.sections[section['id']] = Section(
                        id=section['id'],
                        name=section['name'],
                        year=section['year'],
                        section=section['section'],
                        total_students=section['totalStudents'],
                        class_teacher=section['classTeacher']
                    )
            
            # Load classroom data
            with open(os.path.join(self.input_folder, "classrooms.json"), 'r') as f:
                classrooms_data = json.load(f)
                for classroom in classrooms_data:
                    if classroom['department'] == 'AIML':  # Filter for AIML department
                        self.classrooms[classroom['id']] = Classroom(
                            id=classroom['id'],
                            name=classroom['name'],
                            number=classroom['number'],
                            type=classroom['type'],
                            capacity=classroom['capacity'],
                            department=classroom['department']
                        )
            
            # Load working days
            with open(os.path.join(self.input_folder, "all_semesters_net_dates.json"), 'r') as f:
                semester_data = json.load(f)
                # For demo, use first semester's working days
                self.working_days = semester_data[0]['netDates'] if semester_data else []
                
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
    
    def _estimate_subject_hours(self, subject_name: str) -> int:
        """Estimate hours per week for a subject"""
        lab_subjects = ['lab', 'practical', 'project']
        theory_heavy = ['mathematics', 'physics', 'statistics']
        
        subject_lower = subject_name.lower()
        
        if any(lab in subject_lower for lab in lab_subjects):
            return 4  # Lab subjects typically need more hours
        elif any(theory in subject_lower for theory in theory_heavy):
            return 4  # Theory heavy subjects
        else:
            return 3  # Regular subjects
    
    def _is_lab_subject(self, subject_name: str) -> bool:
        """Determine if a subject is a lab subject"""
        lab_keywords = ['lab', 'practical', 'project', 'programming', 'database', 'networks']
        return any(keyword in subject_name.lower() for keyword in lab_keywords)
    
    def _find_qualified_faculty(self, subject_code: str) -> List[str]:
        """Find faculty members qualified to teach a subject"""
        subject = self.subjects[subject_code]
        qualified_faculty = []
        
        for faculty_id, faculty in self.faculties.items():
            # Check if faculty teaches this subject directly
            if subject.name in faculty.subjects:
                qualified_faculty.append(faculty_id)
                continue
                
            # Check specialization match
            subject_keywords = subject.name.lower().split()
            for spec in faculty.specialization:
                spec_lower = spec.lower()
                if any(keyword in spec_lower for keyword in subject_keywords):
                    qualified_faculty.append(faculty_id)
                    break
        
        return qualified_faculty
    
    def _create_variables(self):
        """Create decision variables for the optimization model"""
        # Main assignment variable: assignment[faculty][subject][section][day][period][classroom]
        self.assignment_vars = {}
        
        # Create variables for each possible assignment
        for faculty_id in self.faculties:
            self.assignment_vars[faculty_id] = {}
            for subject_code in self.subjects:
                # Check if faculty can teach this subject
                if faculty_id not in self._find_qualified_faculty(subject_code):
                    continue
                    
                self.assignment_vars[faculty_id][subject_code] = {}
                for section_id in self.sections:
                    self.assignment_vars[faculty_id][subject_code][section_id] = {}
                    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                        self.assignment_vars[faculty_id][subject_code][section_id][day] = {}
                        for period in range(1, 9):
                            self.assignment_vars[faculty_id][subject_code][section_id][day][period] = {}
                            for classroom_id in self.classrooms:
                                # Check classroom suitability
                                classroom = self.classrooms[classroom_id]
                                section = self.sections[section_id]
                                subject = self.subjects[subject_code]
                                
                                # Skip if classroom capacity is insufficient
                                if classroom.capacity < section.total_students:
                                    continue
                                    
                                # Match classroom type with subject requirements
                                if subject.is_lab and classroom.type != 'lab':
                                    continue
                                if not subject.is_lab and classroom.type == 'lab':
                                    continue
                                
                                var_name = f"assign_{faculty_id}_{subject_code}_{section_id}_{day}_{period}_{classroom_id}"
                                self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id] = \
                                    self.model.NewBoolVar(var_name)
    
    def _add_constraints(self):
        """Add all constraints to the model"""
        self._add_faculty_constraints()
        self._add_classroom_constraints()
        self._add_section_constraints()
        self._add_subject_hour_constraints()
        self._add_lab_constraints()
        self._add_free_period_constraints()
    
    def _add_faculty_constraints(self):
        """Add constraints related to faculty scheduling"""
        for faculty_id in self.faculties:
            faculty = self.faculties[faculty_id]
            
            # Faculty can teach at most one class at a time
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                for period in range(1, 9):
                    assignments = []
                    for subject_code in self.assignment_vars.get(faculty_id, {}):
                        for section_id in self.assignment_vars[faculty_id][subject_code]:
                            for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                assignments.append(
                                    self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                )
                    
                    if assignments:
                        self.model.Add(sum(assignments) <= 1)
            
            # Daily workload constraints
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                daily_assignments = []
                for period in range(1, 9):
                    for subject_code in self.assignment_vars.get(faculty_id, {}):
                        for section_id in self.assignment_vars[faculty_id][subject_code]:
                            for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                daily_assignments.append(
                                    self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                )
                
                if daily_assignments:
                    self.model.Add(sum(daily_assignments) >= faculty.min_hours_per_day)
                    self.model.Add(sum(daily_assignments) <= faculty.max_hours_per_day)
    
    def _add_classroom_constraints(self):
        """Add constraints for classroom utilization"""
        # Each classroom can host only one class at a time
        for classroom_id in self.classrooms:
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                for period in range(1, 9):
                    assignments = []
                    for faculty_id in self.assignment_vars:
                        for subject_code in self.assignment_vars[faculty_id]:
                            for section_id in self.assignment_vars[faculty_id][subject_code]:
                                if classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                    assignments.append(
                                        self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                    )
                    
                    if assignments:
                        self.model.Add(sum(assignments) <= 1)
    
    def _add_section_constraints(self):
        """Add constraints for section scheduling"""
        # Each section can have only one class at a time
        for section_id in self.sections:
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                for period in range(1, 9):
                    assignments = []
                    for faculty_id in self.assignment_vars:
                        for subject_code in self.assignment_vars[faculty_id]:
                            if section_id in self.assignment_vars[faculty_id][subject_code]:
                                for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                    assignments.append(
                                        self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                    )
                    
                    if assignments:
                        self.model.Add(sum(assignments) <= 1)
    
    def _add_subject_hour_constraints(self):
        """Ensure subjects get required hours per week"""
        for subject_code, subject in self.subjects.items():
            for section_id in self.sections:
                section = self.sections[section_id]
                
                # Only assign subjects to appropriate semester sections
                if self._get_section_semester(section) != subject.semester:
                    continue
                
                weekly_assignments = []
                for faculty_id in self.assignment_vars:
                    if subject_code in self.assignment_vars[faculty_id]:
                        if section_id in self.assignment_vars[faculty_id][subject_code]:
                            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                                for period in range(1, 9):
                                    for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                        weekly_assignments.append(
                                            self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                        )
                
                if weekly_assignments:
                    # Ensure minimum hours are scheduled
                    self.model.Add(sum(weekly_assignments) >= subject.hours_per_week)
                    self.model.Add(sum(weekly_assignments) <= subject.hours_per_week + 1)  # Allow slight flexibility
    
    def _add_lab_constraints(self):
        """Add constraints specific to lab sessions"""
        # Lab sessions should be in continuous periods
        for subject_code, subject in self.subjects.items():
            if not subject.is_lab:
                continue
                
            for section_id in self.sections:
                for faculty_id in self.assignment_vars:
                    if subject_code not in self.assignment_vars[faculty_id]:
                        continue
                    if section_id not in self.assignment_vars[faculty_id][subject_code]:
                        continue
                        
                    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                        # If a lab is scheduled in period p, try to schedule next period too
                        for period in range(1, 8):
                            current_assignments = []
                            next_assignments = []
                            
                            for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                current_assignments.append(
                                    self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                )
                            
                            for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period + 1, {}):
                                next_assignments.append(
                                    self.assignment_vars[faculty_id][subject_code][section_id][day][period + 1][classroom_id]
                                )
                            
                            # If current period is scheduled, encourage next period
                            if current_assignments and next_assignments:
                                self.model.Add(sum(current_assignments) <= sum(next_assignments) + 1)
    
    def _add_free_period_constraints(self):
        """Ensure each section has at least one free period per day"""
        for section_id in self.sections:
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                daily_assignments = []
                for period in range(1, 9):
                    for faculty_id in self.assignment_vars:
                        for subject_code in self.assignment_vars[faculty_id]:
                            if section_id in self.assignment_vars[faculty_id][subject_code]:
                                for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day].get(period, {}):
                                    daily_assignments.append(
                                        self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                    )
                
                if daily_assignments:
                    # Ensure at least one free period (max 7 classes per day)
                    self.model.Add(sum(daily_assignments) <= 7)
    
    def _get_section_semester(self, section: Section) -> int:
        """Calculate current semester for a section based on year"""
        # Assuming odd semesters in first half of academic year
        current_month = datetime.now().month
        base_semester = (section.year - 1) * 2
        
        if current_month >= 7:  # July onwards - odd semester
            return base_semester + 1
        else:  # January to June - even semester
            return base_semester + 2
    
    def solve(self) -> bool:
        """Solve the timetable optimization problem"""
        print("Creating variables...")
        self._create_variables()
        
        print("Adding constraints...")
        self._add_constraints()
        
        print("Solving...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300  # 5 minutes timeout
        
        status = solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution found!")
            self._extract_solution(solver)
            return True
        else:
            print("No solution found!")
            return False
    
    def _extract_solution(self, solver):
        """Extract and format the solution"""
        self.timetable = defaultdict(lambda: defaultdict(dict))
        
        for faculty_id in self.assignment_vars:
            for subject_code in self.assignment_vars[faculty_id]:
                for section_id in self.assignment_vars[faculty_id][subject_code]:
                    for day in self.assignment_vars[faculty_id][subject_code][section_id]:
                        for period in self.assignment_vars[faculty_id][subject_code][section_id][day]:
                            for classroom_id in self.assignment_vars[faculty_id][subject_code][section_id][day][period]:
                                var = self.assignment_vars[faculty_id][subject_code][section_id][day][period][classroom_id]
                                if solver.Value(var):
                                    self.timetable[section_id][day][period] = {
                                        'faculty': self.faculties[faculty_id].name,
                                        'subject': self.subjects[subject_code].name,
                                        'classroom': self.classrooms[classroom_id].number,
                                        'faculty_id': faculty_id,
                                        'subject_code': subject_code,
                                        'classroom_id': classroom_id
                                    }
    
    def export_timetable(self, output_file: str = "timetable.json"):
        """Export the generated timetable to JSON"""
        if not hasattr(self, 'timetable'):
            print("No timetable to export. Run solve() first.")
            return
        
        # Convert defaultdict to regular dict for JSON serialization
        export_data = {}
        for section_id in self.timetable:
            export_data[section_id] = {}
            section_name = self.sections[section_id].name
            export_data[section_id]['section_name'] = section_name
            export_data[section_id]['schedule'] = {}
            
            for day in self.timetable[section_id]:
                export_data[section_id]['schedule'][day] = {}
                for period in sorted(self.timetable[section_id][day].keys()):
                    export_data[section_id]['schedule'][day][f"Period_{period}"] = self.timetable[section_id][day][period]
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Timetable exported to {output_file}")
    
    def print_timetable(self, section_id: Optional[str] = None):
        """Print the timetable in a readable format"""
        if not hasattr(self, 'timetable'):
            print("No timetable to display. Run solve() first.")
            return
        
        sections_to_print = [section_id] if section_id else list(self.timetable.keys())
        
        for sec_id in sections_to_print:
            if sec_id not in self.timetable:
                continue
                
            print(f"\n{'='*60}")
            print(f"TIMETABLE FOR {self.sections[sec_id].name}")
            print(f"{'='*60}")
            
            # Print header
            print(f"{'Period':<10}", end="")
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                print(f"{day:<15}", end="")
            print()
            print("-" * 100)
            
            # Print each period
            for period in range(1, 9):
                time_slot = self.time_slots[(period - 1) * 6]  # Get time for this period
                print(f"{period} ({time_slot.start_time})", end=" ")
                
                for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
                    if day in self.timetable[sec_id] and period in self.timetable[sec_id][day]:
                        slot = self.timetable[sec_id][day][period]
                        display = f"{slot['subject'][:8]}({slot['classroom']})"
                        print(f"{display:<15}", end="")
                    else:
                        print(f"{'FREE':<15}", end="")
                print()


def main():
    """Main function to run the timetable generator"""
    print("Initializing Timetable Generator...")
    
    # Create input folder if it doesn't exist
    if not os.path.exists("input"):
        os.makedirs("input")
        print("Created 'input' folder. Please add your JSON files there.")
        return
    
    try:
        generator = TimetableGenerator()
        print(f"Loaded {len(generator.faculties)} faculty members")
        print(f"Loaded {len(generator.subjects)} subjects")
        print(f"Loaded {len(generator.sections)} sections")
        print(f"Loaded {len(generator.classrooms)} classrooms")
        
        if generator.solve():
            generator.print_timetable()
            generator.export_timetable()
        else:
            print("Failed to generate timetable. Try relaxing some constraints.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
