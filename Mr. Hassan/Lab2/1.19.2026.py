"""
Day 3 Activity: Use lambda and .apply() to clean a dataset.
Tasks:
1) Clean price (remove $ and commas)
2) Fill missing quantity with 0
3) Create total = price * quantity
4) Categorize price: low / med / high
"""

import pandas as pd

# Sample dataset
raw = {
    "product": ["Widget A", "Widget B", "Widget C"],
    "price": ["$1,234.50", "$567.89", "$2,345.00"],
    "quantity": [10, 5, None],
}

df = pd.DataFrame(raw)

# TODO: Clean price column using .apply with lambda.
# TODO: Fill missing quantity with 0.
# TODO: Create total column.
# TODO: Add price_category column (low/med/high).

# print(df)


df = pd.DataFrame(raw)
df["price"] = df["price"].apply(lambda x: float(x.replace("$", "").replace(",", "")))
df["quantity"] = df["quantity"].fillna(0)
df["total"] = df["price"] * df["quantity"]
df["price_category"] = df["price"].apply(
    lambda x: "low" if x < 600 else "med" if x < 2000 else "high"
)

print(df)
"""
Day 4 Activity: Parse nested dictionaries (student database).
Tasks:
1) Get Alice's AI301 grade
2) Calculate Bob's GPA (weighted by credits)
3) Find all students in CS101
4) Get average grade across all courses
5) Find student with highest GPA
"""

students = {
    "S001": {
        "name": "Alice Chen",
        "courses": {
            "CS101": {"grade": 92, "credits": 3},
            "MATH201": {"grade": 88, "credits": 4},
            "AI301": {"grade": 95, "credits": 3},
        },
        "advisor": "Dr. Smith",
    },
    "S002": {
        "name": "Bob Lee",
        "courses": {
            "CS101": {"grade": 85, "credits": 3},
            "MATH201": {"grade": 90, "credits": 4},
        },
        "advisor": "Dr. Patel",
    },
}

# TODO: Implement the tasks above using nested dict access.
alice_ai301 = students["S001"]["courses"]["AI301"]["grade"]
bob_courses = students["S002"]["courses"]
bob_total_points = sum(c["grade"] * c["credits"] for c in bob_courses.values())
bob_total_credits = sum(c["credits"] for c in bob_courses.values())
bob_gpa = bob_total_points / bob_total_credits
cs101_students = [
    s["name"]
    for s in students.values()
    if "CS101" in s["courses"]
]
all_grades = [
    course["grade"]
    for student in students.values()
    for course in student["courses"].values()
]
avg_grade = sum(all_grades) / len(all_grades)
def calc_gpa(student):
    courses = student["courses"].values()
    total = sum(c["grade"] * c["credits"] for c in courses)
    credits = sum(c["credits"] for c in courses)
    return total / credits
highest_gpa_student = max(students.values(), key=calc_gpa)["name"]
print("Alice AI301:", alice_ai301)
print("Bob GPA:", round(bob_gpa, 2))
print("CS101 students:", cs101_students)
print("Average grade:", round(avg_grade, 2))
print("Top GPA student:", highest_gpa_student)

