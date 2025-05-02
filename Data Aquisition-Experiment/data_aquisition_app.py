import sys
import os
import csv
import pygame
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import PyPDF2
import serial
import time
import logging
import re
from datetime import datetime, timezone

# Mapping for subscripts and superscripts
_sub_map = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_sup_map = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")

def format_math_text(text):
    """
    Convert plain text formulas so that digits after letters become subscripts
    and ^n sequences become superscripts, e.g. H2O → H₂O, x^2 → x².
    """
    # superscripts from ^n patterns
    text = re.sub(r"\^([0-9+\-=\(\)])",
                  lambda m: m.group(1).translate(_sup_map), text)
    # subscripts for digits after letters or closing parenthesis
    text = re.sub(r"([A-Za-z\)])([0-9+\-=\(\)])",
                  lambda m: m.group(1) + m.group(2).translate(_sub_map), text)
    return text
from tobiiEyeTrackerPro import TobiiEyeTracker

TRIGGER_READING  = 0x10
TRIGGER_THINKING = 0x20
TRACKING_THREAD  = False

# --- synchronisation helpers ---
EXP_START_NS   = None          # set once in main()
TRIGGER_EVENTS = []            # [(abs_time_s, rel_time_s, label, code)]


def send_trigger(trigger_port, trigger_code, label: str = ""):
    """
    Hardware trigger + in‑memory log with absolute & relative times.

    The relative clock is time since EXP_START_NS in seconds.
    """
    global EXP_START_NS, TRIGGER_EVENTS
    t_ns   = time.perf_counter_ns()
    abs_s  = t_ns / 1e9
    rel_s  = (t_ns - EXP_START_NS) / 1e9 if EXP_START_NS else 0.0

    # pulse
    if trigger_port is not None:
        try:
            trigger_port.write([trigger_code])
            time.sleep(0.004)
            trigger_port.write([0x00])
        except Exception as e:
            print("Error sending trigger:", e)

    # store for later CSV merge
    TRIGGER_EVENTS.append((abs_s, rel_s, label, trigger_code))


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
def select_mode_screen(screen, font):
    """
    Ask user to choose Developer or Experiment mode.
    Returns True for experiment mode, False for developer mode.
    """
    # Dynamically center the buttons on the screen
    screen_width = screen.get_width()
    screen_height = screen.get_height()
    button_width, button_height = 200, 60
    spacing = 50
    total_width = button_width * 2 + spacing
    start_x = (screen_width - total_width) // 2
    start_y = screen_height // 2
    dev_rect = pygame.Rect(start_x, start_y, button_width, button_height)
    exp_rect = pygame.Rect(start_x + button_width + spacing, start_y, button_width, button_height)
    while True:
        for event in pygame.event.get():
            # Exit the app if Q is pressed or window is closed.
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if dev_rect.collidepoint(event.pos):
                    return False
                if exp_rect.collidepoint(event.pos):
                    return True
        screen.fill((255,255,255))
        title = font.render("Select Mode", True, (0,0,0))
        screen.blit(title, ((screen.get_width()-title.get_width())//2, 200))
        draw_button(screen, dev_rect, "Developer", font)
        draw_button(screen, exp_rect, "Experiment", font)
        pygame.display.flip()

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
    # apply subscript/superscript formatting
    text = format_math_text(text)
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
            send_trigger(trigger_port, TRIGGER_READING, "ReadingStart")
            trigger_sent = True


# -----------------------------------------
# 4c. Displaying Each Question with Checkboxes
# -----------------------------------------
def question_screen(screen, font, question_dict, user_name, trigger_port):
    checkbox_size = 20
    spacing_y = 50
    options = list(question_dict["options"].items())

    # format question text for math symbols
    q_text = format_math_text(question_dict["question"])
    question_surface = font.render(q_text, True, (0, 0, 0))
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

    # Set the y-position for the buttons
    submit_y = start_y + question_height + gap1 + options_height + gap2
    # Calculate dynamic widths for "Submit" and "Frustration"
    submit_width = 100
    frustration_text = "Frustration"
    frustration_width = font.size(frustration_text)[0] + 20  # padding around text
    button_height = 50
    spacing = 50  # space between buttons
    total_width = frustration_width + spacing + submit_width
    start_x = (screen.get_width() - total_width) // 2
    # Place Frustration button on the left, Submit on the right
    frustration_button_rect = pygame.Rect(start_x, submit_y, frustration_width, button_height)
    submit_button_rect = pygame.Rect(start_x + frustration_width + spacing, submit_y, submit_width, button_height)
    frustration_flag = False
    # Precompute left-aligned column X positions
    cols = 2
    col_centers = [screen.get_width() // 4, 3 * screen.get_width() // 4]
    col_max_widths = [0] * cols
    for idx, (letter, option_text) in enumerate(options):
        fmt = format_math_text(option_text)
        opt_width = font.size(f"{letter}: {fmt}")[0]
        total_opt = checkbox_size + 10 + opt_width
        col = idx % cols
        if total_opt > col_max_widths[col]:
            col_max_widths[col] = total_opt
    col_start_x = [col_centers[i] - col_max_widths[i] // 2 for i in range(cols)]

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
                # Check grid-based option collisions (2 columns, left-aligned)
                row_spacing = spacing * 2
                for idx, (letter, option_text) in enumerate(options):
                    row = idx // cols
                    col = idx % cols
                    x = col_start_x[col]
                    y = start_y + question_height + gap1 + row * row_spacing
                    cb_rect = pygame.Rect(x, y, checkbox_size, checkbox_size)
                    if cb_rect.collidepoint(mouse_pos):
                        # Toggle selection
                        if selected_option == letter:
                            selected_option = None
                        else:
                            selected_option = letter
                # Frustration button click (toggle only if not already flagged)
                if frustration_button_rect.collidepoint(mouse_pos) and not frustration_flag:
                    frustration_flag = True
                # Submit button click (only if an option selected)
                if submit_button_rect.collidepoint(mouse_pos) and selected_option is not None:
                    return selected_option, frustration_flag

        if not trigger_sent:
            send_trigger(trigger_port, TRIGGER_THINKING, "QuestionStart")
            trigger_sent = True

        screen.fill((255, 255, 255))
        draw_user_name(screen, font, user_name)

        # Render and center the question text
        question_surface = font.render(q_text, True, (0, 0, 0))
        q_x = (screen.get_width() - question_surface.get_width()) // 2
        screen.blit(question_surface, (q_x, start_y))

        # Render each option in a grid (2 columns, left-aligned)
        row_spacing = spacing * 2
        for idx, (letter, option_text) in enumerate(options):
            row = idx // cols
            col = idx % cols
            x = col_start_x[col]
            y = start_y + question_height + gap1 + row * row_spacing
            cb_rect = pygame.Rect(x, y, checkbox_size, checkbox_size)
            pygame.draw.rect(screen, (0, 0, 0), cb_rect, 2)
            if selected_option == letter:
                pygame.draw.line(screen, (0, 0, 0), (cb_rect.left, cb_rect.top), (cb_rect.right, cb_rect.bottom), 2)
                pygame.draw.line(screen, (0, 0, 0), (cb_rect.left, cb_rect.bottom), (cb_rect.right, cb_rect.top), 2)
            fmt = format_math_text(option_text)
            opt_surface = font.render(f"{letter}: {fmt}", True, (0, 0, 0))
            screen.blit(opt_surface, (x + checkbox_size + 10, y))

        # Draw Frustration button: red until clicked, then gray
        if frustration_flag:
            draw_button(screen, frustration_button_rect, "Frustration", font, button_color=(150, 150, 150))
        else:
            draw_button(screen, frustration_button_rect, "Frustration", font, button_color=(200, 100, 100))
        # Draw Submit button on the right
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
            # Exit the app if Q is pressed or window is closed.
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
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


def start_tracking(eye_tracker, save_dir, experiment_mode):
    # ensure ET_data subdirectory exists next to user CSV
    et_dir = os.path.join(save_dir, "ET_data")
    os.makedirs(et_dir, exist_ok=True)
    file_path = os.path.join(et_dir, "eye_tracking_data.csv")
    if eye_tracker is None:
        eye_tracker = TobiiEyeTracker(et_dir)
    else:
        eye_tracker.save_dir = et_dir
    # Connect only if not already connected

    if eye_tracker.exp_start_ns is None:
        eye_tracker.set_exp_start_ns(EXP_START_NS)

    if eye_tracker.et is None:
        if experiment_mode:
            while not eye_tracker.connect():
                time.sleep(1)
        else:
            try:
                eye_tracker.connect()
            except Exception:
                pass
    eye_tracker.start_recording()
    interval = 1  # seconds

    # Open CSV file and write headers
    with open(file_path, mode='w', newline='') as file:
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

        print(f"Eye Tracking data saved to {file_path}")

    # --- graceful shutdown & binary export -------------------------------
    eye_tracker.stop_recording()
    eye_tracker.save_data_npy(et_dir)


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
    # (moved to after user name input)

    # 4. Now initialize Pygame.
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Reading Comprehension Quiz")
    # Allow OS access by not grabbing input
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    # Use a Unicode-capable font for subscripts/superscripts
    pygame.font.init()
    font_path = pygame.font.match_font("dejavusans") or pygame.font.match_font("arial") or pygame.font.get_default_font()
    font = pygame.font.Font(font_path, 24)

    # 5. Initialize trigger port.
    try:
        trigger_port = serial.Serial("COM4")  # Adjust COM port if needed
        trigger_port.write([0x00])
    except Exception as e:
        print("Error opening trigger port:", e)
        trigger_port = None

    # Mode selection
    experiment_mode = select_mode_screen(screen, font)
    # Ensure eye variable exists even in developer mode
    eye = None

    if experiment_mode:
        # ensure ET_data subdirectory exists next to user CSV
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "test_data")
        et_dir = os.path.join(save_dir, "ET_data")
        os.makedirs(et_dir, exist_ok=True)
        file_path = os.path.join(et_dir, "eye_tracking_data.csv")
        # wait up to 30s for eye-tracker
        eye = TobiiEyeTracker(et_dir)
        start_time = time.time()
        timeout = 10
        connected = False
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        while time.time() - start_time < timeout:
            if eye.connect():
                connected = True
                break
            # draw waiting screen with progress bar
            elapsed = time.time() - start_time
            progress = elapsed / timeout
            screen.fill((255,255,255))
            msg = font.render("Waiting for Eye-Tracker...", True, (0,0,0))
            x = (screen_width - msg.get_width())//2
            y = screen_height//2 - 40
            screen.blit(msg, (x, y))
            # progress bar
            bar_width = screen_width * 0.6
            bar_height = 20
            bar_x = (screen_width - bar_width)//2
            bar_y = y + 40
            pygame.draw.rect(screen, (0,0,0), (bar_x, bar_y, bar_width, bar_height), 2)
            pygame.draw.rect(screen, (0,150,0), (bar_x+2, bar_y+2, (bar_width-4)*progress, bar_height-4))
            pygame.display.flip()
            time.sleep(0.1)
        # brief pause after connect or timeout
        time.sleep(0.5)

    # 6. Ask for the user's name.
    user_name = input_name_screen(screen, font)

    # Create user data directory and start eye-tracking thread
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_dir = os.path.join(base_dir, "experiment_data", user_name)
    os.makedirs(user_data_dir, exist_ok=True)

    # ---------- logs directory ----------
    log_dir = os.path.join(user_data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # human‑readable run log
    run_log_path = os.path.join(log_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(run_log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("==== Session started ====")


    # Set experiment start time for high-res trigger logging
    global EXP_START_NS
    EXP_START_NS = time.perf_counter_ns()

    # -----------------------------------------------------------------
    # Create or reuse a single TobiiEyeTracker instance for the session
    # -----------------------------------------------------------------
    if eye is None:
        eye = TobiiEyeTracker()          # dev‑mode fallback
    # inject high‑resolution experiment start time
    eye.set_exp_start_ns(EXP_START_NS)

    import threading
    tracking_thread = threading.Thread(
        target=start_tracking,
        args=(eye, user_data_dir, experiment_mode),
        daemon=True
    )
    tracking_thread.start()

    results = []  # This list will store tuples: (Section #, Question #, Correct, User Answer, Frustration)
    total_sections = len(sections)

    # 7. Loop through all reading comprehension sections.
    for sec_index, section in enumerate(sections):
        # Display the reading comprehension passage and send trigger for reading.
        reading_comprehension_screen(screen, font, section["paragraph"], user_name, trigger_port)
        # Loop through all questions in this section.
        for q_index, question in enumerate(section["questions"]):
            answer, frustration = question_screen(screen, font, question, user_name, trigger_port)

            submit_ns = time.perf_counter_ns()
            abs_s = submit_ns / 1e9  # numeric seconds
            rel_s = (submit_ns - EXP_START_NS) / 1e9
            abs_iso = datetime.fromtimestamp(abs_s, timezone.utc).isoformat()

            results.append((sec_index + 1, q_index + 1, question["correct"], answer, frustration, abs_iso, f"{rel_s:.6f}", "QuizSubmit"))
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

    # 9. Write results and triggers to a CSV file.
    directory = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(directory, "experiment_data", f"{user_name}"), exist_ok=True)
    csv_filename = os.path.join(directory, "experiment_data", f"{user_name}", f"{user_name}-data.csv")
    header = [
              "Section", "Question",
              "Correct", "Answer", "Frustration",
              "abs_time_s", "rel_time_s", "Label"]

    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

            # quiz answer rows
            for sec, qn, corr, ans, frustr in results:
                writer.writerow(["quiz",
                                 sec, qn,
                                 corr, ans, frustr,
                                 "", "", ""])

            # trigger rows
            for abs_s, rel_s, label, _code in TRIGGER_EVENTS:
                iso = datetime.fromtimestamp(abs_s, timezone.utc).isoformat()
                writer.writerow(["", "", "", "", "",
                                 iso, f"{rel_s:.6f}", label])
        print(f"Results saved to {csv_filename}")
    except Exception as e:
        print("Error writing CSV file:", e)

    # 10. Close trigger port and quit.
    if trigger_port is not None:
        trigger_port.close()

    # note shutdown
    logging.info("==== Session finished ====")

    pygame.quit()
    global TRACKING_THREAD
    TRACKING_THREAD = True
    tracking_thread.join()   # wait for the tracking thread to write .npy files


if __name__ == "__main__":
    # Run on main thread (required for macOS GUI), disable input grab to allow OS access
    main()