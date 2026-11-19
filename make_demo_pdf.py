"""Creates a SYNTHETIC demo PDF (made-up rules, NOT a real college document) for a smoke test.
Needs reportlab only for this script:  pip install reportlab"""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from src.config import PATHS

PAGES = [
 ("Attendance Rules", "Students must maintain a minimum attendance of 75 percent in each subject to be eligible for the end semester examination. Students with attendance between 65 and 75 percent may apply for condonation by paying a condonation fee of 500 rupees and submitting a medical certificate to the head of department. Students with attendance below 65 percent are not eligible to appear in the examination and must repeat the course in the next academic year. Attendance is recorded for every lecture, practical and tutorial session by the course teacher and displayed on the notice board every month."),
 ("Examination Guidelines", "The end semester examination carries 70 marks and the internal assessment carries 30 marks for every theory course. A student must score at least 40 percent marks in the end semester examination and 40 percent in total to pass a course. Students must carry the college identity card to the examination hall. Mobile phones and smart watches are strictly prohibited in the examination hall. Students caught using unfair means will be reported to the examination malpractice committee and may be debarred for one semester."),
 ("Academic Calendar 2025-26", "The odd semester begins on 4 August 2025 and the first internal test is held in the week starting 15 September 2025. The Diwali vacation runs from 20 October to 26 October 2025. The odd semester end examination begins on 1 December 2025. The even semester begins on 12 January 2026 and the even semester end examination begins on 4 May 2026. The summer vacation starts on 1 June 2026."),
 ("Revaluation and Grading", "A student may apply for revaluation of an answer sheet within 7 days of the declaration of results by paying a fee of 800 rupees per paper. The result of revaluation is declared within 21 days. Grades are awarded on a ten point scale. A grade of O is awarded for 80 percent marks and above, and grade F means the student has failed the course. The Semester Grade Point Average is calculated as the weighted average of grade points using course credits as weights."),
]
doc = SimpleDocTemplate(str(PATHS["raw"] / "DEMO_synthetic_college_rules.pdf"), pagesize=A4)
st = getSampleStyleSheet()
story = []
for i, (title, body) in enumerate(PAGES):
    story += [Paragraph(title, st["Heading1"]), Spacer(1, 12), Paragraph(body, st["BodyText"])]
    if i < len(PAGES) - 1:
        story.append(PageBreak())
doc.build(story)
print("Created demo PDF in", PATHS["raw"])
