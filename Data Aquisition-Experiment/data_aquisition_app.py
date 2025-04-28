import sys
import os
import csv
import pygame
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import PyPDF2
import serial
import time
import threading
from tobiiEyeTrackerPro import TobiiEyeTracker

# Trigger constants and helper function
TRIGGER_READING = 0x10
TRIGGER_THINKING = 0x20
TRACKING_THREAD = False

# TODO: Adjust the experiment to auto calculate the interface position to display everything in center

def send_trigger(trigger_port, trigger_code):
    if trigger_port is not None:
        try:
            trigger_port.write([trigger_code])
            time.sleep(0.01)
            trigger_port.write([0x00])
        except Exception as e:
            print("Error sending trigger:", e)


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
# 3. Parsing the Reading Comprehensions
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
    # Center the input box and button on screen
    screen_width = screen.get_width()
    screen_height = screen.get_height()
    input_box_width, input_box_height = 600, 50
    input_box_x = (screen_width - input_box_width) // 2
    input_box_y = (screen_height - input_box_height) // 2 - 50
    input_box = pygame.Rect(input_box_x, input_box_y, input_box_width, input_box_height)
    next_button_width, next_button_height = 100, 50
    next_button_x = (screen_width - next_button_width) // 2
    next_button_y = input_box_y + input_box_height + 20
    next_button_rect = pygame.Rect(next_button_x, next_button_y, next_button_width, next_button_height)
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
        # Center the prompt above the input box
        prompt_surface = font.render("Enter your name:", True, (0, 0, 0))
        prompt_x = (screen_width - prompt_surface.get_width()) // 2
        prompt_y = input_box_y - prompt_surface.get_height() - 10
        screen.blit(prompt_surface, (prompt_x, prompt_y))
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
        x = (screen.get_width() - rendered_line.get_width()) // 2
        screen.blit(rendered_line, (x, y))
        y += font.get_linesize() + 5


def reading_comprehension_screen(screen, font, passage, user_name, trigger_port):
    next_button_rect = pygame.Rect((screen.get_width()-100)//2, 500, 100, 50)
    trigger_sent = False
    while True:
        for event in pygame.event.get():
            # Exit the app if Q is pressed or window is closed.
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if next_button_rect.collidepoint(event.pos):
                    return

        screen.fill((255, 255, 255))
        draw_user_name(screen, font, user_name)
        display_text(screen, font, passage, start_y=150)
        draw_button(screen, next_button_rect, "Next", font)
        pygame.display.flip()
        if not trigger_sent:
            send_trigger(trigger_port, TRIGGER_READING)
            trigger_sent = True


# -----------------------------------------
# 4c. Displaying Each Question with Checkboxes
# -----------------------------------------
def question_screen(screen, font, question_dict, user_name, trigger_port):
    checkbox_size = 20
    spacing_y = 50
    options = list(question_dict["options"].items())

    # Render the question text once to get its height
    question_surface = font.render(question_dict["question"], True, (0, 0, 0))
    question_height = question_surface.get_height()

    # Calculate total height for options block
    options_height = len(options) * spacing_y

    # Define additional gaps and submit button height
    gap1 = 60  # gap between question and options
    gap2 = 30  # gap between options and submit button
    submit_button_height = 50  # height of the submit button

    # Total height of the entire questions view block
    total_height = question_height + gap1 + options_height + gap2 + submit_button_height

    # Calculate the starting y-position to vertically center the block
    start_y = (screen.get_height() - total_height) // 2

    # Set the y-position for the submit button
    submit_y = start_y + question_height + gap1 + options_height + gap2
    submit_button_rect = pygame.Rect((screen.get_width() - 100) // 2, submit_y, 100, 50)

    trigger_sent = False
    selected_option = None

    while True:
        for event in pygame.event.get():
            # Exit the app if Q is pressed or window is closed.
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                # Check each option's checkbox collision
                for idx, (letter, option_text) in enumerate(options):
                    opt_surface = font.render(f"{letter}: {option_text}", True, (0, 0, 0))
                    row_width = checkbox_size + 10 + opt_surface.get_width()
                    row_x = (screen.get_width() - row_width) // 2
                    option_y = start_y + question_height + gap1 + idx * spacing_y
                    cb_rect = pygame.Rect(row_x, option_y, checkbox_size, checkbox_size)
                    if cb_rect.collidepoint(mouse_pos):
                        selected_option = letter
                if submit_button_rect.collidepoint(mouse_pos) and selected_option is not None:
                    return selected_option

        if not trigger_sent:
            send_trigger(trigger_port, TRIGGER_THINKING)
            trigger_sent = True

        screen.fill((255, 255, 255))
        draw_user_name(screen, font, user_name)

        # Render and center the question text
        question_surface = font.render(question_dict["question"], True, (0, 0, 0))
        q_x = (screen.get_width() - question_surface.get_width()) // 2
        screen.blit(question_surface, (q_x, start_y))

        # Render each option, centering each row horizontally
        for idx, (letter, option_text) in enumerate(options):
            opt_surface = font.render(f"{letter}: {option_text}", True, (0, 0, 0))
            row_width = checkbox_size + 10 + opt_surface.get_width()
            row_x = (screen.get_width() - row_width) // 2
            option_y = start_y + question_height + gap1 + idx * spacing_y
            cb_rect = pygame.Rect(row_x, option_y, checkbox_size, checkbox_size)
            pygame.draw.rect(screen, (0, 0, 0), cb_rect, 2)
            if selected_option == letter:
                pygame.draw.line(screen, (0, 0, 0), (cb_rect.left, cb_rect.top), (cb_rect.right, cb_rect.bottom), 2)
                pygame.draw.line(screen, (0, 0, 0), (cb_rect.left, cb_rect.bottom), (cb_rect.right, cb_rect.top), 2)
            screen.blit(opt_surface, (cb_rect.right + 10, cb_rect.top))

        draw_button(screen, submit_button_rect, "Submit", font)
        pygame.display.flip()


# -----------------------------------------
# 4d. Transition Screen Between Sections
# -----------------------------------------
def show_section_transition(screen, font, user_name, current_section, total_sections):
    next_button_rect = pygame.Rect((screen.get_width()-100)//2, 500, 100, 50)
    message = f"Section {current_section} complete. Click Next for Section {current_section + 1} of {total_sections}."
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
        x_msg = (screen.get_width() - msg_surface.get_width()) // 2
        screen.blit(msg_surface, (x_msg, 300))
        draw_button(screen, next_button_rect, "Next", font)
        pygame.display.flip()


def start_tracking():
    eye_tracker = TobiiEyeTracker()
    eye_tracker.connect()  # TODO: make it so if you get False (meaning eyetracker is not there), exit the app
    eye_tracker.start_recording()
    interval = 1  # seconds

    # Open CSV file and write headers
    with open('eye_tracking_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Mean Pupil Area', 'Mean Fixation Duration', 'User Looking', 'Avg X', 'Avg Y'])

        while not TRACKING_THREAD:
            time.sleep(interval)
            mean_pupil_area = eye_tracker.get_mean_pupil_area(interval)
            mean_fixation_duration = eye_tracker.get_mean_fixation_duration(interval)
            user_looking = eye_tracker.is_user_looking(interval)
            avg_x, avg_y = eye_tracker.get_average_gaze_point(interval)
            eye_tracker.cleanup_old_data(interval)

            # Write a row of data with a timestamp
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                mean_pupil_area,
                mean_fixation_duration,
                user_looking,
                avg_x,
                avg_y
            ])


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

    # 3. Start the eye-tracker in a separate thread.
    tracking_thread = threading.Thread(target=start_tracking, daemon=True)
    tracking_thread.start()

    # 4. Now initialize Pygame.
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Reading Comprehension Quiz")
    font = pygame.font.SysFont("Arial", 24)

    # 5. Initialize trigger port.
    try:
        trigger_port = serial.Serial("COM4")  # Adjust COM port if needed
        trigger_port.write([0x00])
    except Exception as e:
        print("Error opening trigger port:", e)
        trigger_port = None

    # 6. Ask for the user's name.
    user_name = input_name_screen(screen, font)

    results = []  # This list will store tuples: (Section #, Question #, Correct, User Answer)
    total_sections = len(sections)

    # 7. Loop through all reading comprehension sections.
    for sec_index, section in enumerate(sections):
        # Display the reading comprehension passage and send trigger for reading.
        reading_comprehension_screen(screen, font, section["paragraph"], user_name, trigger_port)
        # Loop through all questions in this section.
        for q_index, question in enumerate(section["questions"]):
            answer = question_screen(screen, font, question, user_name, trigger_port)
            results.append((sec_index + 1, q_index + 1, question["correct"], answer))
        # If not the last section, show a transition screen.
        if sec_index < total_sections - 1:
            show_section_transition(screen, font, user_name, sec_index + 1, total_sections)

    # 8. Display a completion message.
    screen.fill((255, 255, 255))
    draw_user_name(screen, font, user_name)
    thanks_text = f"Quiz complete! Thank you, {user_name}."
    thanks_surface = font.render(thanks_text, True, (0, 0, 0))
    x_thanks = (screen.get_width() - thanks_surface.get_width()) // 2
    screen.blit(thanks_surface, (x_thanks, 300))
    pygame.display.flip()
    pygame.time.wait(3000)

    # 9. Write results to a CSV file.
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

    # 10. Close trigger port and quit.
    if trigger_port is not None:
        trigger_port.close()
    pygame.quit()
    global TRACKING_THREAD
    TRACKING_THREAD = True


if __name__ == "__main__":
    main()