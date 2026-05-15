import os
import re
from pathlib import Path


# Folder that contains the raw AWS Lambda markdown documentation.
SOURCE_ROOT = Path("data/aws-lambda-developer-guide-main")


# Folder where the cleaned .txt files will be written.
OUTPUT_ROOT = Path("clean_data")


# Lines that often indicate shell commands or other noisy setup text.
SHELL_PREFIXES = (
    "$ ",
    "# ",
    "> ",
    "pip ",
    "npm ",
    "yarn ",
    "git ",
    "aws ",
    "sam ",
    "docker ",
    "cd ",
    "mkdir ",
    "rm ",
    "curl ",
    "wget ",
)


def looks_like_shell_command(line):
    """Return True when a line looks like a command rather than explanation."""
    stripped = line.lstrip()
    lower = stripped.lower()
    return lower.startswith(SHELL_PREFIXES)


def looks_like_json_or_config(line):
    """Return True when a line looks like JSON, YAML, or config dump text."""
    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith(("{", "}", "[", "]")):
        return True

    if stripped.endswith(("{", "}", "[", "]")):
        return True

    brace_count = stripped.count("{") + stripped.count("}")
    quote_count = stripped.count('"')
    colon_count = stripped.count(":")
    equals_count = stripped.count("=")

    if brace_count >= 2 and colon_count >= 2:
        return True

    if quote_count >= 4 and colon_count >= 2:
        return True

    if equals_count >= 4 and len(stripped) > 80:
        return True

    if colon_count >= 6 and len(stripped) > 120:
        return True

    if len(stripped) > 180 and (colon_count >= 3 or equals_count >= 3):
        return True

    return False


def clean_inline_code(text):
    """Remove backticks but keep the useful words inside them."""
    return re.sub(r"`([^`]+)`", r"\1", text)


def remove_markdown_images(text):
    """Remove markdown image syntax like ![alt text](url)."""
    return re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)


def extract_markdown_link_text(text):
    """Extract text from markdown links, e.g., [text](url) becomes text."""
    return re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)


def looks_like_version_boilerplate(line):
    """Return True if line is mostly version/dependency noise."""
    stripped = line.strip()

    if len(stripped) < 100:
        version_pattern = r"v\d+\.\d+|version\s*\d+|\d+\.\d+\.\d+|newer|older|require|depend"
        version_count = len(re.findall(version_pattern, stripped, re.IGNORECASE))

        if version_count >= 2:
            return True

    return False


def clean_markdown_text(text):
    """Clean markdown text line by line and keep the useful explanation."""
    cleaned_lines = []
    in_code_block = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        # Skip fenced code blocks completely.
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Drop empty lines for now; we will normalize spacing later.
        if not stripped:
            cleaned_lines.append("")
            continue

        # Remove lines that are mostly commands or config dumps.
        if looks_like_shell_command(stripped):
            continue

        if looks_like_json_or_config(stripped):
            continue

        # Skip lines that are pure markdown image syntax.
        if stripped.startswith("!["):
            continue

        # Skip version/dependency boilerplate.
        if looks_like_version_boilerplate(stripped):
            continue

        # Keep headings and normal prose, but clean up markdown.
        line = remove_markdown_images(line)
        line = extract_markdown_link_text(line)
        line = clean_inline_code(line)

        # Remove long repeated punctuation that often comes from copied output.
        line = re.sub(r"[-=_]{4,}", "", line)

        # Trim extra spaces after removing markdown.
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    # Collapse repeated blank lines into a single blank line.
    final_lines = []
    previous_blank = False

    for line in cleaned_lines:
        is_blank = not line.strip()

        if is_blank:
            if not previous_blank:
                final_lines.append("")
            previous_blank = True
        else:
            final_lines.append(line)
            previous_blank = False

    return "\n".join(final_lines).strip() + "\n"


def process_file(input_path):
    """Read one markdown file, clean it, and write the text output."""
    relative_path = input_path.relative_to(SOURCE_ROOT)
    output_path = OUTPUT_ROOT / relative_path.with_suffix(".txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="replace") as file_handle:
        raw_text = file_handle.read()

    cleaned_text = clean_markdown_text(raw_text)

    with output_path.open("w", encoding="utf-8", errors="replace") as file_handle:
        file_handle.write(cleaned_text)

    return output_path


def main():
    """Find all markdown files, clean them, and save the results."""
    if not SOURCE_ROOT.exists():
        print(f"Source folder not found: {SOURCE_ROOT}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    markdown_files = sorted(SOURCE_ROOT.rglob("*.md"))
    print(f"Found {len(markdown_files)} markdown files")

    for index, input_path in enumerate(markdown_files, start=1):
        output_path = process_file(input_path)
        print(f"[{index}/{len(markdown_files)}] Cleaned {input_path} -> {output_path}")

    print(f"Finished. Cleaned files are in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()