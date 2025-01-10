import re
from src.utils import save_json, logger

# Load the file content
with open("data/RT_textbook.txt", "r") as file:
    content = file.read()

# Split content into paragraphs separated by "\n\n"
paragraphs = content.split("\n\n")


# Function to extract paragraphs containing equations and their context
def extract_equation_context(paragraphs):
    equation_pattern = re.compile(r".*\s=\s.*")  # Regex to detect equations
    extracted_content = []

    for idx, paragraph in enumerate(paragraphs):
        if equation_pattern.search(paragraph):  # Detect if the paragraph contains " = "
            # Get previous, current, and next paragraphs
            previous_paragraph = (
                paragraphs[idx - 1].strip() if idx > 0 else "No previous paragraph"
            )
            next_paragraph = (
                paragraphs[idx + 1].strip()
                if idx + 1 < len(paragraphs)
                else "No next paragraph"
            )
            extracted_content.append(
                previous_paragraph + "\n\n" + paragraph + "\n\n" + next_paragraph
            )

    return extracted_content


# Extract paragraphs with equations and their context
all_content = extract_equation_context(paragraphs)

# Save results to JSON
save_json(all_content, fname="generated/extracted_rag.json")

logger.info(f"Extracted {len(all_content)} entries with equations.")
