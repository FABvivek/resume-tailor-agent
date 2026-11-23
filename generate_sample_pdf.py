"""
Generates a realistic sample candidate resume PDF for testing and demoing.
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_sample_pdf(filename: str = "sample_resume.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Alex Mercer")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 68, "alex.mercer@email.com | github.com/alexmercer | (555) 234-5678")

    # Education
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 100, "EDUCATION")
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.line(50, height - 105, width - 50, height - 105)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 120, "University of Washington")
    c.setFont("Helvetica", 10)
    c.drawString(width - 170, height - 120, "Graduation: June 2025")
    c.drawString(50, height - 135, "B.S. in Computer Science, GPA: 3.8/4.0")

    # Skills
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 165, "TECHNICAL SKILLS")
    c.line(50, height - 170, width - 50, height - 170)

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 185, "Languages: Python, SQL, C++, TypeScript, Bash")
    c.drawString(50, height - 200, "Frameworks & Backend: FastAPI, Flask, SQLAlchemy, Docker, Redis")
    c.drawString(50, height - 215, "AI & Data: PyTorch, Sentence Transformers, scikit-learn, Pandas, SQLite")

    # Projects
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 245, "PROJECTS & EXPERIENCE")
    c.line(50, height - 250, width - 50, height - 250)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 270, "AI Career Search Assistant | Python, FastAPI, Docker, SQLite")
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 285, "• Designed and deployed a REST API serving RAG queries over 1,500 internship postings.")
    c.drawString(60, height - 300, "• Built vector embedding search using MiniLM embeddings, achieving sub-50ms query response.")
    c.drawString(60, height - 315, "• Containerized application using Docker and docker-compose for reproducible local environments.")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 340, "Distributed Task Scheduler | Python, Redis, SQL")
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 355, "• Built a lightweight distributed job queue backed by Redis with worker retry logic.")
    c.drawString(60, height - 370, "• Implemented SQL schema migrations and transaction rollback tests with pytest.")
    c.drawString(60, height - 385, "• Profiled memory bottlenecks and cut worker memory consumption during heavy loads.")

    c.save()
    print(f"Created sample PDF at {filename}")


if __name__ == "__main__":
    create_sample_pdf()
