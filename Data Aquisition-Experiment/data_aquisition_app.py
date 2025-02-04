import sys
import os
import csv
import pygame
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import PyPDF2

# -------------------------------
# 1. File Selection via Tkinter
# -------------------------------
def select_pdf_file():
    # Hide the main Tkinter window
    Tk().withdraw()
    filename = askopenfilename(title="Select PDF file", filetypes=[("PDF files", "*.pdf")])
    return filename

# ------------------------------------
# 2. PDF Text Extraction with PyPDF2
# ------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("Error reading PDF:", e)
    return text

# ----------------------------------
# 3. Parsing the Reading Comprehension
# ----------------------------------
def parse_reading_comprehensions(raw_text):
    """
    Parses the raw text and returns a list of dictionaries.
    Each dictionary represents one reading comprehension section with:
      - title: e.g. "Reading Comprehension 1"
      - paragraph: the passage text
      - questions: a list of questions; each question is a dict with:
            - question: the question text
            - options: a dict mapping option letters (A, B, C, etc.) to text
            - correct: the correct answer (as letter) [this is stored but not shown]
    """
    sections = []
    current_section = None
    lines = raw_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # New section begins when we find a line starting with "Reading Comprehension"
        if line.startswith("Reading Comprehension"):
            if current_section:
                sections.append(current_section)
            current_section = {"title": line, "paragraph": "", "questions": []}
            i += 1
            continue

        # Paragraph block (starts with "• Paragraph:")
        if line.startswith("• Paragraph:"):
            paragraph_lines = [line.replace("• Paragraph:", "").strip()]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("– Question:"):
                paragraph_lines.append(lines[i].strip())
                i += 1
            current_section["paragraph"] = " ".join(paragraph_lines)
            continue

        # Question block (starts with "– Question:")
        if line.startswith("– Question:"):
            question_text = line.replace("– Question:", "").strip()
            options = {}
            correct = None
            i += 1
            while i < len(lines):
                subline = lines[i].strip()
                if subline.startswith("— Option"):
                    try:
                        option_line = subline.replace("— Option", "").strip()
                        letter, option_text = option_line.split(":", 1)
                        options[letter.strip()] = option_text.strip()
                    except Exception as e:
                        print("Error parsing option:", subline, e)
                elif subline.startswith("— Correct Answer:"):
                    correct = subline.replace("— Correct Answer:", "").strip()
                elif subline.startswith("– Question:") or subline.startswith("Reading Comprehension"):
                    break
                i += 1
            current_section["questions"].append({
                "question": question_text,
                "options": options,
                "correct": correct
            })
            continue

        i += 1

    if current_section:
        sections.append(current_section)
    return sections

# -----------------------------------------
# Helper functions for drawing UI elements
# -----------------------------------------
def draw_button(screen, rect, text, font, button_color=(100, 200, 100), text_color=(0, 0, 0)):
    pygame.draw.rect(screen, button_color, rect)
    rendered_text = font.render(text, True, text_color)
    text_rect = rendered_text.get_rect(center=rect.center)
    screen.blit(rendered_text, text_rect)

def draw_user_name(screen, font, user_name):
    """ Draws the user's name at the top right corner. """
    if user_name:
        name_surface = font.render(user_name, True, (0, 0, 0))
        x_pos = screen.get_width() - name_surface.get_width() - 10
        screen.blit(name_surface, (x_pos, 10))

# -----------------------------------------
# 4a. User Name Input Screen (After PDF selection)
# -----------------------------------------
def input_name_screen(screen, font):
    name = ""
    input_box = pygame.Rect(100, 200, 600, 50)
    next_button_rect = pygame.Rect(350, 300, 100, 50)
    active = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        if name.strip():
                            return name
                    elif event.key == pygame.K_BACKSPACE:
                        name = name[:-1]
                    else:
                        name += event.unicode
            if event.type == pygame.MOUSEBUTTONDOWN:
                if next_button_rect.collidepoint(event.pos) and name.strip():
                    return name

        screen.fill((255, 255, 255))
        prompt_surface = font.render("Enter your name:", True, (0, 0, 0))
        screen.blit(prompt_surface, (100, 150))
        txt_surface = font.render(name, True, (0, 0, 0))
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        pygame.draw.rect(screen, (0, 0, 0), input_box, 2)
        draw_button(screen, next_button_rect, "Next", font)
        pygame.display.flip()

# -----------------------------------------
# 4b. Displaying the Reading Comprehension Passage
# -----------------------------------------
def display_text(screen, font, text, start_y=50, color=(0, 0, 0)):
    words = text.split()
    lines = []
    current_line = ""
    max_width = screen.get_width() - 100  # margin on both sides

    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    if current_line:
        lines.append(current_line)

    y = start_y
    for line in lines:
        rendered_line = font.render(line, True, color)
        screen.blit(rendered_line, (50, y))
        y += font.get_linesize() + 5

def reading_comprehension_screen(screen, font, passage, user_name):
    next_button_rect = pygame.Rect(350, 500, 100, 50)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if next_button_rect.collidepoint(event.pos):
                    return

        screen.fill((255, 255, 255))
        draw_user_name(screen, font, user_name)
        display_text(screen, font, passage, start_y=50)
        draw_button(screen, next_button_rect, "Next", font)
        pygame.display.flip()

# -----------------------------------------
# 4c. Displaying Each Question with Checkboxes
# -----------------------------------------
def question_screen(screen, font, question_dict, user_name):
    submit_button_rect = pygame.Rect(350, 500, 100, 50)
    base_y = 150
    checkbox_size = 20
    spacing_y = 50
    selected_option = None
    options = list(question_dict["options"].items())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for idx, (letter, option_text) in enumerate(options):
                    cb_rect = pygame.Rect(100, base_y + idx * spacing_y, checkbox_size, checkbox_size)
                    if cb_rect.collidepoint(mouse_pos):
                        selected_option = letter
                if submit_button_rect.collidepoint(mouse_pos) and selected_option is not None:
                    return selected_option

        screen.fill((255, 255, 255))
        draw_user_name(screen, font, user_name)
        question_surface = font.render(question_dict["question"], True, (0, 0, 0))
        screen.blit(question_surface, (50, 50))
        for idx, (letter, option_text) in enumerate(options):
            cb_rect = pygame.Rect(100, base_y + idx * spacing_y, checkbox_size, checkbox_size)
            pygame.draw.rect(screen, (0, 0, 0), cb_rect, 2)
            if selected_option == letter:
                pygame.draw.line(screen, (0, 0, 0), (cb_rect.left, cb_rect.top), (cb_rect.right, cb_rect.bottom), 2)
                pygame.draw.line(screen, (0, 0, 0), (cb_rect.left, cb_rect.bottom), (cb_rect.right, cb_rect.top), 2)
            opt_surface = font.render(f"{letter}: {option_text}", True, (0, 0, 0))
            screen.blit(opt_surface, (cb_rect.right + 10, cb_rect.top))
        draw_button(screen, submit_button_rect, "Submit", font)
        pygame.display.flip()

# -----------------------------------------
# 4d. Transition Screen Between Sections
# -----------------------------------------
def show_section_transition(screen, font, user_name, current_section, total_sections):
    next_button_rect = pygame.Rect(350, 500, 100, 50)
    message = f"Section {current_section} complete. Click Next for Section {current_section+1} of {total_sections}."
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if next_button_rect.collidepoint(event.pos):
                    return
        screen.fill((255, 255, 255))
        draw_user_name(screen, font, user_name)
        msg_surface = font.render(message, True, (0, 0, 0))
        screen.blit(msg_surface, (50, 300))
        draw_button(screen, next_button_rect, "Next", font)
        pygame.display.flip()

# ----------------------------
# Main application loop
# ----------------------------
def main():
    # 1. Let the user select a PDF file BEFORE initializing Pygame.
    pdf_path = select_pdf_file()
    if not pdf_path:
        print("No file selected.")
        return

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print("No text extracted from the PDF.")
        return

    # 2. Parse the text into reading comprehension sections.
    sections = parse_reading_comprehensions(raw_text)
    if not sections:
        print("No reading comprehension sections found.")
        return

    # 3. Now initialize Pygame.
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Reading Comprehension Quiz")
    font = pygame.font.SysFont("Arial", 24)

    # 4. Ask for the user's name.
    user_name = input_name_screen(screen, font)

    results = []  # This list will store tuples: (Section #, Question #, Correct, User Answer)
    total_sections = len(sections)

    # 5. Loop through all reading comprehension sections.
    for sec_index, section in enumerate(sections):
        # Display the reading comprehension passage.
        reading_comprehension_screen(screen, font, section["paragraph"], user_name)
        # Loop through all questions in this section.
        for q_index, question in enumerate(section["questions"]):
            answer = question_screen(screen, font, question, user_name)
            results.append( (sec_index+1, q_index+1, question["correct"], answer) )
        # If not the last section, show a transition screen.
        if sec_index < total_sections - 1:
            show_section_transition(screen, font, user_name, sec_index+1, total_sections)

    # 6. Display a completion message.
    screen.fill((255, 255, 255))
    draw_user_name(screen, font, user_name)
    thanks_text = f"Quiz complete! Thank you, {user_name}."
    thanks_surface = font.render(thanks_text, True, (0, 0, 0))
    screen.blit(thanks_surface, (200, 300))
    pygame.display.flip()
    pygame.time.wait(3000)

    # 7. Write results to a CSV file in the same directory as the code.
    directory = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(directory, f"{user_name}-data.csv")
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Reading comprehension number", "Question number", "Correct answer", "User chosen answer"])
            for row in results:
                writer.writerow(row)
        print(f"Results saved to {csv_filename}")
    except Exception as e:
        print("Error writing CSV file:", e)

    pygame.quit()

if __name__ == "__main__":
    main()